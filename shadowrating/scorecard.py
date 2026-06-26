"""
Phase 2 / Model A: rules-based scorecard.

The "rules" are entirely in Phase 1 -- fixed pillar weights from config,
applied to percentile-rank-scaled indicators. This module's only job is to put
the unitless [0, 1] composite onto the agency notch scale (1..21) so it can be
compared to actual ratings, and to do that calibration honestly.

The one fitted piece is a single-variable least-squares line (notch ~ a +
b*composite). That is a unit conversion, not a learned weighting model --
the pillar weights themselves stay fixed and config-driven. Model B (Phase 3)
is where an actual statistical model (ordered logit) gets to choose weights.

Per CLAUDE.md principle 1, the headline number is the leave-one-country-out
(LOOCV) error, not the in-sample fit -- refitting the line on 41 countries and
predicting the 42nd is cheap and removes the "fit the line to the same points
you're scoring it against" bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ratings as ratings_mod


def fit_linear_calibration(composite: pd.Series, notch: pd.Series) -> tuple[float, float]:
    """Least-squares (intercept, slope) for notch ~ intercept + slope * composite."""
    mask = composite.notna() & notch.notna()
    x = composite[mask].to_numpy(dtype=float)
    y = notch[mask].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return intercept, slope


def predict_notch(composite: pd.Series, intercept: float, slope: float) -> pd.Series:
    raw = intercept + slope * composite
    return raw.clip(lower=1, upper=21)


def loocv_predict(composite: pd.Series, notch: pd.Series) -> pd.Series:
    """
    Leave-one-country-out predictions: for each country with both a composite
    score and an actual notch, refit the calibration line on every other such
    country and predict the held-out one. Countries missing either value get
    NaN -- never silently scored against an in-sample fit.
    """
    mask = composite.notna() & notch.notna()
    idx = composite.index[mask]
    out = pd.Series(np.nan, index=composite.index, dtype=float)

    for held_out in idx:
        train = idx.drop(held_out)
        intercept, slope = fit_linear_calibration(composite.loc[train], notch.loc[train])
        out.loc[held_out] = float(
            predict_notch(pd.Series([composite.loc[held_out]]), intercept, slope).iloc[0]
        )
    return out


def build_scorecard(pillar_scores: pd.DataFrame, composite: pd.Series,
                     ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the full per-country scorecard: pillar scores, composite,
    in-sample predicted notch, LOOCV (out-of-sample) predicted notch, the
    actual consensus notch, and the divergence -- the actual analytical
    product of this whole phase.
    """
    ratings_idx = ratings_df.set_index("iso3")
    notch = ratings_idx["consensus_notch"].astype(float).reindex(composite.index)

    intercept, slope = fit_linear_calibration(composite, notch)
    in_sample_pred = predict_notch(composite, intercept, slope)
    oos_pred = loocv_predict(composite, notch)

    table = pillar_scores.copy()
    table["composite"] = composite
    table["actual_notch"] = notch
    table["actual_letter"] = notch.map(ratings_mod.notch_to_sp_letter)
    table["pred_notch"] = in_sample_pred
    table["pred_letter"] = in_sample_pred.round().map(ratings_mod.notch_to_sp_letter)
    table["loocv_pred_notch"] = oos_pred
    table["loocv_letter"] = oos_pred.round().map(ratings_mod.notch_to_sp_letter)
    table["divergence"] = (table["loocv_pred_notch"] - table["actual_notch"]).round(1)
    table["name"] = ratings_idx["name"].reindex(composite.index)

    table.attrs["calibration"] = {"intercept": intercept, "slope": slope}
    return table


def error_summary(table: pd.DataFrame) -> dict:
    """LOOCV error stats -- the headline numbers, not in-sample fit."""
    err = table["loocv_pred_notch"] - table["actual_notch"]
    err = err.dropna()

    ig_boundary = ratings_mod.IG_BOUNDARY_NOTCH
    actual_ig = table["actual_notch"] >= ig_boundary
    pred_ig = table["loocv_pred_notch"] >= ig_boundary
    both = actual_ig.notna() & pred_ig.notna()
    ig_mismatches = table.loc[both & (actual_ig != pred_ig)]

    return {
        "n": len(err),
        "mae": err.abs().mean(),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": err.mean(),
        "ig_mismatch_count": len(ig_mismatches),
        "ig_mismatches": ig_mismatches,
    }
