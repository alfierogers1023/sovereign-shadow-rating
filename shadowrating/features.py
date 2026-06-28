"""
Phase 1: scale the indicator panel and aggregate into pillar features.

Scaling = signed percentile rank -- more robust to this data's fat tails than
raw z-scores. Direction-adjusted so higher always means "more creditworthy"
after scaling, regardless of the indicator's raw sign.

Scaling is sample-relative: ranks are computed against whichever countries are
in the panel for this run. They will not transfer to a country outside the
sample without recomputing ranks against the new sample -- this is a known
limitation, not a bug.

`loocv_folds` below exists for a related but distinct reason: scaling against
the *full* panel (as `build_features` does) means a held-out country's
percentile rank is computed using a distribution that includes itself, which
leaks (mild, X-only, not target leakage, but real and measured -- mean 3.2pp,
max 4.7pp shift per indicator across all 42 countries) feature-engineering
information into Phase 2/3's LOOCV. The properly fold-wise version recomputes
the scaler on the 41 training countries only, then ranks the held-out
country's raw values against that training distribution alone. `build_features`
is still what every "descriptive" use of pillar scores should call (e.g. the
dashboard's pillar bar chart for a single country) -- it's `scorecard.py` and
`model_b.py`'s LOOCV that need the fold-wise version.
"""
from __future__ import annotations

import numpy as np
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
        mean of whatever it has -- imputed transparently within the pillar,
        made explicit via missing_counts rather than silently zero-filling.
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


def _percentile_against(value: float, train_values: pd.Series, direction: int) -> float:
    """
    Percentile rank of `value` against `train_values` only -- `value` itself
    is never part of the distribution it's ranked against. Uses mid-rank
    (ties split the difference) so this matches pandas' default
    `rank(pct=True)` convention when `value` happens to equal a training
    value, keeping the fold-wise and full-sample scalers comparable.
    """
    if pd.isna(value):
        return np.nan
    tv = train_values.dropna()
    if tv.empty:
        return np.nan
    less = (tv < value).sum()
    equal = (tv == value).sum()
    pct = (less + 0.5 * equal) / len(tv)
    return pct if direction >= 0 else 1.0 - pct


def scale_held_out(row: pd.Series, train_panel: pd.DataFrame) -> pd.Series:
    """
    Percentile-rank-scale one country's raw indicator row against a training
    panel only -- the fold-wise counterpart to `scale_panel` for a single
    held-out observation.
    """
    meta = indicator_meta()
    out = {}
    for col in row.index:
        direction = meta.get(col, {}).get("direction", 1)
        out[col] = _percentile_against(row[col], train_panel[col], direction)
    return pd.Series(out, index=row.index)


def loocv_folds(panel: pd.DataFrame):
    """
    Yield, for every country in `panel`, a fully fold-wise rebuild of Phase 1:

        (held_out_iso3, train_pillar_scores, train_composite,
         held_out_pillar_scores, held_out_composite)

    `train_*` are computed from `scale_panel` run on the other countries
    only; `held_out_*` are the held-out country's pillar score / composite,
    built by ranking its raw indicators against that same 41-country
    training distribution -- it never sees its own value when computing its
    own rank. This is what `scorecard.loocv_predict` and
    `model_b.fit_*_loocv` consume so the LOOCV error estimate isn't
    contaminated by full-sample scaling. Recomputing `scale_panel` on a
    41-country panel 42 times is cheap (rank computations on ~18 columns);
    don't reach for caching here without a measured reason to.
    """
    for held_out in panel.index:
        train_panel = panel.drop(held_out)
        train_feats = build_features(train_panel)

        held_scaled = scale_held_out(panel.loc[held_out], train_panel)
        held_pillar_df, _, _ = pillar_scores(pd.DataFrame([held_scaled], index=[held_out]))
        held_composite = composite_score(held_pillar_df)

        yield (
            held_out,
            train_feats["pillar_scores"],
            train_feats["composite"],
            held_pillar_df.loc[held_out],
            held_composite.loc[held_out],
        )
