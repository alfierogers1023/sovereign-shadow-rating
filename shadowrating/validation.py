"""
Phase 4: validation -- combine Model A and Model B into one master divergence
table, and cross-check against market-implied risk (FRED bond-yield spreads)
where that data is available.

This is the analytical payload the whole project is built around (see
CLAUDE.md principle 1): not "is the model accurate" but "where do the model(s)
and the agencies disagree, and is there independent evidence (the market)
that the model is onto something." Everything here is LOOCV / out-of-sample --
never in-sample fit.
"""
from __future__ import annotations

import pandas as pd

from . import ratings as ratings_mod


def build_master_table(scorecard_table: pd.DataFrame, band_table: pd.DataFrame,
                       gbm_preds: pd.Series, ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per country: actual rating, Model A's LOOCV prediction + divergence,
    Model B's two LOOCV predictions (ordered-logit band, GBM notch) +
    divergence, and an agreement flag for whether both models diverge from the
    agencies in the same direction (the strongest version of the signal).
    """
    rt_idx = ratings_df.set_index("iso3")
    out = pd.DataFrame(index=scorecard_table.index)
    out["name"] = rt_idx["name"].reindex(out.index)
    out["actual_notch"] = rt_idx["consensus_notch"].astype(float).reindex(out.index)
    out["actual_letter"] = out["actual_notch"].map(ratings_mod.notch_to_sp_letter)

    out["model_a_pred_notch"] = scorecard_table["loocv_pred_notch"]
    out["model_a_divergence"] = (out["model_a_pred_notch"] - out["actual_notch"]).round(1)

    out["model_b_gbm_pred_notch"] = gbm_preds.reindex(out.index)
    out["model_b_gbm_divergence"] = (out["model_b_gbm_pred_notch"] - out["actual_notch"]).round(1)

    out["model_b_band_pred"] = band_table["band_pred"].reindex(out.index)
    out["model_b_band_true"] = band_table["band_true"].reindex(out.index)
    out["model_b_band_divergence"] = out["model_b_band_pred"] - out["model_b_band_true"]

    # Uncertainty, not just a point estimate: the ordered logit's own 90%
    # prediction interval (model_b.fit_ordered_logit_loocv). A country is
    # only "confidently divergent" by this measure if its actual band falls
    # *outside* the model's own stated uncertainty -- stricter than, and a
    # check on, the fixed->2-notch point-estimate flag below. Point estimates
    # are kept (model_b_band_pred above), this is in addition, not instead.
    out["band_ci_lower"] = band_table["band_ci_lower"].reindex(out.index)
    out["band_ci_upper"] = band_table["band_ci_upper"].reindex(out.index)
    out["outside_ci"] = band_table["outside_ci"].reindex(out.index)

    # Agreement: both models' notch-scale divergences are material (>2
    # notches -- roughly "at least one broad letter category") and point the
    # same direction. >0.5 would flag most of the sample and dilute the
    # signal; 2 notches keeps this to genuinely large, consistent gaps.
    a = out["model_a_divergence"]
    b = out["model_b_gbm_divergence"]
    out["models_agree_direction"] = (
        (a.abs() > 2) & (b.abs() > 2) & (((a > 0) & (b > 0)) | ((a < 0) & (b < 0)))
    )

    # The strongest version of the signal: both point-estimate models agree
    # AND the ordered logit's own uncertainty says the actual rating is
    # outside its 90% interval -- not just "the point guess differs," but
    # "the model is confident the agencies are wrong," corroborated by a
    # second, differently-built model's point estimate too.
    out["confident_divergence"] = out["models_agree_direction"] & out["outside_ci"].fillna(False)

    out["max_abs_divergence"] = out[["model_a_divergence", "model_b_gbm_divergence"]].abs().max(axis=1)
    return out


def cross_check_market_spread(master: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame | None:
    """
    Merge FRED spread-vs-benchmark data and report, for the FRED-covered
    subset only, whether each model's divergence direction is corroborated or
    contradicted by the market: a model that rates a country *worse* than the
    agencies (negative divergence) is corroborated if that country's spread is
    *wider* than its agency notch alone would predict, and vice versa.

    Returns None if no FRED data is available (key not set / no coverage) --
    callers should treat that as "skipped," not "zero correlation."

    Coverage caveat: FRED's IRLTLT01 series is OECD-only, so this almost never
    reaches the distressed end of the sample where divergence is largest --
    the cross-check is a useful sanity check on rich-country pricing, not a
    full validation.
    """
    if yields is None or yields.empty:
        return None

    merged = master.join(yields.set_index("iso3"), how="inner")
    if merged.empty or merged["spread_vs_benchmark_bps"].isna().all():
        return None

    # What spread would the agency notch alone predict? Simple linear fit,
    # spread ~ a + b * actual_notch, fit on this FRED-covered subset only.
    import numpy as np
    x = merged["actual_notch"].to_numpy(dtype=float)
    y = merged["spread_vs_benchmark_bps"].to_numpy(dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    merged["spread_resid_bps"] = merged["spread_vs_benchmark_bps"] - (intercept + slope * merged["actual_notch"])

    # Corroborated: model divergence and spread residual point the same way --
    # model says "worse than agency" (negative) and market also prices more
    # risk than the agency notch implies (positive residual), or vice versa.
    merged["model_a_corroborated"] = (
        (merged["model_a_divergence"] < 0) & (merged["spread_resid_bps"] > 0)
    ) | (
        (merged["model_a_divergence"] > 0) & (merged["spread_resid_bps"] < 0)
    )

    cols = ["name", "actual_letter", "model_a_divergence", "model_b_gbm_divergence",
            "spread_vs_benchmark_bps", "spread_resid_bps", "model_a_corroborated"]
    return merged[cols].sort_values("spread_resid_bps", ascending=False)
