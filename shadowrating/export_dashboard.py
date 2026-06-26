"""
Phase 6: export everything the static Chart.js dashboard needs as plain JSON.

GitHub Pages serves static files only -- no Python backend. So this module
re-runs the full pipeline (cheap: LOOCV over 42 countries takes seconds) and
writes flat JSON into docs/data/, which docs/app.js fetches client-side. The
dashboard itself never imports shadowrating; this is the one-way bridge.

Re-running rather than reading cached parquet keeps the dashboard's numbers
guaranteed consistent with a single source of truth at export time, rather
than silently mixing whatever happened to be cached from different runs.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, ratings as ratings_mod
from .loaders import worldbank, imf_weo, fred
from . import coverage as coverage_mod
from . import features as features_mod
from . import data_quality as data_quality_mod
from . import scorecard as scorecard_mod
from . import model_b as model_b_mod
from . import validation as validation_mod
from . import dsa as dsa_mod

DOCS_DATA_DIR = config.PROJECT_ROOT / "docs" / "data"


def _round(obj, ndigits=3):
    """Recursively round floats so the JSON stays small and diffable."""
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, ndigits) for v in obj]
    return obj


def _write_json(name: str, payload) -> None:
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DATA_DIR / name
    with open(path, "w") as f:
        json.dump(_round(payload), f, indent=1)
    print(f"[export] {path.relative_to(config.PROJECT_ROOT)} "
          f"({path.stat().st_size / 1024:.1f} KB)")


def build_all(use_cache: bool = True) -> None:
    rt = ratings_mod.load_ratings()
    rt_idx = rt.set_index("iso3")

    frames = [worldbank.fetch_wdi(use_cache), worldbank.fetch_wgi(use_cache),
              imf_weo.fetch_weo(use_cache)]
    panel = coverage_mod.assemble_panel(*frames)
    vintage = coverage_mod.assemble_vintage(*frames)
    cov = coverage_mod.coverage_report(panel)

    feats = features_mod.build_features(panel)
    pillar_scores, composite = feats["pillar_scores"], feats["composite"]
    notch = rt_idx["consensus_notch"].astype(float).reindex(pillar_scores.index)

    scorecard_table = scorecard_mod.build_scorecard(panel, pillar_scores, composite, rt)
    a_summary = scorecard_mod.error_summary(scorecard_table)

    band_table = model_b_mod.fit_ordered_logit_loocv(panel, notch)
    band_summary = model_b_mod.band_error_summary(band_table)
    gbm_preds, gbm_importance = model_b_mod.fit_gbm_loocv(panel, notch)
    gbm_summary = model_b_mod.notch_error_summary(notch, gbm_preds)

    master = validation_mod.build_master_table(scorecard_table, band_table, gbm_preds, rt)

    dq = data_quality_mod.build_data_quality_table(vintage, feats["missing_counts"],
                                                    total_indicators=feats["scaled"].shape[1])
    master = master.join(dq)

    yields = fred.fetch_yields(use_cache)
    cross = validation_mod.cross_check_market_spread(master, yields)

    # -- countries.json: one row per country, everything the table + detail
    # view needs in one place. --
    countries = []
    for iso3 in master.index:
        row = {
            "iso3": iso3,
            "name": rt_idx.loc[iso3, "name"],
            "actual_notch": master.loc[iso3, "actual_notch"],
            "actual_letter": master.loc[iso3, "actual_letter"],
            "sp": rt_idx.loc[iso3, "sp"], "fitch": rt_idx.loc[iso3, "fitch"],
            "moodys": rt_idx.loc[iso3, "moodys"],
            "ig_flag": bool(rt_idx.loc[iso3, "ig_flag"]),
            "split": int(rt_idx.loc[iso3, "split"]) if pd.notna(rt_idx.loc[iso3, "split"]) else None,
            "composite": composite.get(iso3),
            "pillars": {p: pillar_scores.loc[iso3, p] for p in pillar_scores.columns
                       if iso3 in pillar_scores.index},
            "model_a_pred_notch": master.loc[iso3, "model_a_pred_notch"],
            "model_a_divergence": master.loc[iso3, "model_a_divergence"],
            "model_b_gbm_pred_notch": master.loc[iso3, "model_b_gbm_pred_notch"],
            "model_b_gbm_divergence": master.loc[iso3, "model_b_gbm_divergence"],
            "model_b_band_pred": band_table.loc[iso3, "band_pred"] if iso3 in band_table.index else None,
            "band_ci_lower": master.loc[iso3, "band_ci_lower"],
            "band_ci_upper": master.loc[iso3, "band_ci_upper"],
            "band_proba": [band_table.loc[iso3, f"band_proba_{i}"] for i in range(model_b_mod.N_BANDS)]
                if iso3 in band_table.index and pd.notna(band_table.loc[iso3, "band_proba_0"]) else None,
            "outside_ci": bool(master.loc[iso3, "outside_ci"]) if pd.notna(master.loc[iso3, "outside_ci"]) else None,
            "models_agree_direction": bool(master.loc[iso3, "models_agree_direction"]),
            "confident_divergence": bool(master.loc[iso3, "confident_divergence"]),
            "max_excess_age_years": master.loc[iso3, "max_excess_age_years"],
            "stale_pillars": master.loc[iso3, "stale_pillars"],
            "missing_share": master.loc[iso3, "missing_share"],
        }
        countries.append(row)
    _write_json("countries.json", countries)

    # -- summary.json: headline numbers for the dashboard's top panel. --
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_countries": len(rt),
        "n_ig": int(rt["ig_flag"].sum()),
        "coverage_overall_fill_rate": cov["overall_fill_rate"],
        "model_a": {"mae": a_summary["mae"], "rmse": a_summary["rmse"],
                    "bias": a_summary["bias"], "ig_mismatch_count": a_summary["ig_mismatch_count"]},
        "model_b_ordered_logit": {"exact_match_rate": band_summary["exact_match_rate"],
                                  "within_one_band_rate": band_summary["within_one_band_rate"],
                                  "mae_bands": band_summary["mae_bands"]},
        "model_b_gbm": {"mae": gbm_summary["mae"], "rmse": gbm_summary["rmse"],
                        "bias": gbm_summary["bias"]},
        "gbm_feature_importance": gbm_importance.to_dict(),
        "agreement_count": int(master["models_agree_direction"].sum()),
        "confident_divergence_count": int(master["confident_divergence"].sum()),
        "ci_calibration": {
            "target_mass": model_b_mod.CI_MASS,
            "empirical_coverage": float(1 - band_table["outside_ci"].dropna().astype(bool).mean()),
            "n": int(band_table["outside_ci"].notna().sum()),
        },
        "data_quality": {
            "countries_with_stale_outlier": int((master["max_excess_age_years"] > 0).sum()),
            "mean_missing_share": float(master["missing_share"].mean()),
        },
        "market_cross_check": (
            {"n": len(cross), "corroborated": int(cross["model_a_corroborated"].sum())}
            if cross is not None else None
        ),
    }
    _write_json("summary.json", summary)

    # -- dsa.json: scenarios + fan chart, nested by country for compact fetch. --
    panel_for_dsa = panel  # raw levels, not percentile-scaled
    scenarios_long, fan_long = dsa_mod.build_dsa_tables(panel_for_dsa)
    frag = dsa_mod.fragility_summary(scenarios_long)

    dsa_payload = {}
    for iso3 in scenarios_long["iso3"].unique():
        sc = scenarios_long[scenarios_long["iso3"] == iso3]
        scenarios = {
            name: sc[sc["scenario"] == name].sort_values("year_ahead")["debt_gdp"].tolist()
            for name in sc["scenario"].unique()
        }
        fan = fan_long[fan_long["iso3"] == iso3].sort_values("year_ahead")
        dsa_payload[iso3] = {
            "years": sc[sc["scenario"] == "baseline"].sort_values("year_ahead")["year_ahead"].tolist(),
            "scenarios": scenarios,
            "fan": {col: fan[col].tolist() for col in fan.columns if col.startswith("p")},
        }
    _write_json("dsa.json", dsa_payload)

    frag_payload = {iso3: row.to_dict() for iso3, row in frag.iterrows()}
    _write_json("dsa_fragility.json", frag_payload)

    print(f"\nExported {len(countries)} countries -> docs/data/. "
          f"Open docs/index.html (via a local server, not file://) to view.")
