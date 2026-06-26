"""
Phase 1: scale the indicator panel and aggregate into pillar features.

Scaling = signed percentile rank (decision in CLAUDE.md -- more robust to this
data's fat tails than raw z-scores). Direction-adjusted so higher always means
"more creditworthy" after scaling, regardless of the indicator's raw sign.

Scaling is sample-relative: ranks are computed against whichever countries are
in the panel for this run. They will not transfer to a country outside the
sample without recomputing ranks against the new sample -- this is a known
limitation, not a bug.
"""
from __future__ import annotations

import pandas as pd

from . import config


def indicator_meta() -> dict[str, dict]:
    """Friendly indicator name -> {pillar, direction}, across all three sources."""
    meta: dict[str, dict] = {}
    for d in (config.WDI_INDICATORS, config.WGI_INDICATORS, config.WEO_INDICATORS):
        for v in d.values():
            meta[v["name"]] = {"pillar": v["pillar"], "direction": v["direction"]}
    return meta


def scale_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Signed percentile rank per indicator, in [0, 1], higher = more creditworthy.
    NaNs stay NaN here -- this step only rescales what's present and never
    imputes (imputation, where it happens, happens at the pillar-aggregation
    step below, and is documented there).
    """
    meta = indicator_meta()
    scaled = pd.DataFrame(index=panel.index)
    for col in panel.columns:
        direction = meta.get(col, {}).get("direction", 1)
        pct = panel[col].rank(pct=True, na_option="keep")
        scaled[col] = pct if direction >= 0 else 1.0 - pct
    return scaled


def pillar_scores(scaled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Average the scaled indicators within each pillar, per country.

    Returns (scores, missing_counts, empty_pillars):
      - scores: country x pillar, mean of available scaled indicators. A
        country missing some (but not all) of a pillar's indicators gets the
        mean of whatever it has -- this is the "impute transparently within
        pillar" rule from CLAUDE.md, made explicit via missing_counts rather
        than silently zero-filling.
      - missing_counts: country x pillar, how many of that pillar's indicators
        were absent and excluded from the average.
      - empty_pillars: pillars with zero indicators anywhere in this panel
        (e.g. "fiscal" until the WEO loader has data) -- these are dropped
        entirely rather than reported as 100% missing, since there's nothing
        to impute from.
    """
    meta = indicator_meta()
    all_pillars = sorted(set(m["pillar"] for m in meta.values()))
    cols_by_pillar = {
        p: [c for c in scaled.columns if meta.get(c, {}).get("pillar") == p]
        for p in all_pillars
    }

    scores = pd.DataFrame(index=scaled.index)
    missing = pd.DataFrame(index=scaled.index)
    empty_pillars = []
    for p, cols in cols_by_pillar.items():
        if not cols:
            empty_pillars.append(p)
            continue
        sub = scaled[cols]
        scores[p] = sub.mean(axis=1, skipna=True)
        missing[p] = sub.isna().sum(axis=1)
    return scores, missing, empty_pillars


def composite_score(pillar_df: pd.DataFrame) -> pd.Series:
    """
    Weighted composite across whichever pillars actually have data this run,
    renormalising config.PILLAR_WEIGHTS over the available set -- and, within
    a row, over whichever pillars that specific country has a value for. A
    pillar missing for one country never silently counts as zero; it's just
    excluded and the other weights are scaled up to fill the gap.
    """
    available = [p for p in pillar_df.columns if p in config.PILLAR_WEIGHTS]
    w = pd.Series({p: config.PILLAR_WEIGHTS[p] for p in available})

    weighted = pillar_df[available].mul(w, axis=1)
    row_weight = pillar_df[available].notna().mul(w, axis=1).sum(axis=1)
    return weighted.sum(axis=1, skipna=True) / row_weight


def build_features(panel: pd.DataFrame) -> dict:
    """Run the full Phase 1 pipeline and return everything worth inspecting."""
    scaled = scale_panel(panel)
    scores, missing, empty_pillars = pillar_scores(scaled)
    composite = composite_score(scores)
    return {
        "scaled": scaled,
        "pillar_scores": scores,
        "missing_counts": missing,
        "empty_pillars": empty_pillars,
        "composite": composite,
    }
