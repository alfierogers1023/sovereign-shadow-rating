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

LOOCV here goes one level deeper than just refitting the calibration line:
`loocv_predict` takes the *raw panel*, not a precomputed composite, and uses
`features.loocv_folds` to rebuild the percentile-rank scaling itself on the
41 training countries each fold. Calibrating against a composite that was
scaled using the full 42-country sample (including the held-out point) would
leak feature-engineering information into a number we report as out-of-sample
-- see CLAUDE.md's validation-integrity note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as features_mod
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


def loocv_predict(panel: pd.DataFrame, notch: pd.Series) -> pd.Series:
    """
    Leave-one-country-out predictions, fold-wise from the raw panel: for each
    country, rebuild percentile-rank scaling on the other 41 (via
    `features.loocv_folds`), fit the calibration line on their composites vs.
    their actual notches, and predict the held-out country's notch from its
    own fold-scaled composite. Countries missing a target or composite get
    NaN -- never silently scored against an in-sample (or leaked-scaling) fit.
    """
    out = pd.Series(np.nan, index=panel.index, dtype=float)

    for held_out, _, train_composite, _, held_composite in features_mod.loocv_folds(panel):
        if pd.isna(notch.get(held_out)) or pd.isna(held_composite):
            continue
        train_notch = notch.reindex(train_composite.index)
        fit_mask = train_composite.notna() & train_notch.notna()
        if fit_mask.sum() < 2:
            continue
        intercept, slope = fit_linear_calibration(
            train_composite.loc[fit_mask], train_notch.loc[fit_mask]
        )
        out.loc[held_out] = float(
            predict_notch(pd.Series([held_composite]), intercept, slope).iloc[0]
        )
    return out


def build_scorecard(panel: pd.DataFrame, pillar_scores: pd.DataFrame, composite: pd.Series,
                     ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble the full per-country scorecard: pillar scores, composite,
    in-sample predicted notch, LOOCV (out-of-sample, fold-wise-scaled)
    predicted notch, the actual consensus notch, and the divergence -- the
    actual analytical product of this whole phase.

    `pillar_scores`/`composite` (full-sample-scaled, from `features.build_features`)
    are used only for display (the composite column, the in-sample
    calibration line) -- never for the validated `loocv_pred_notch`, which is
    rebuilt fold-wise from `panel` by `loocv_predict`.
    """
    ratings_idx = ratings_df.set_index("iso3")
    notch = ratings_idx["consensus_notch"].astype(float).reindex(composite.index)

    intercept, slope = fit_linear_calibration(composite, notch)
    in_sample_pred = predict_notch(composite, intercept, slope)
    oos_pred = loocv_predict(panel, notch)

    table = pillar_scores.copy()
    table["composite"] = composite
    table["actual_notch"] = notch
    table["actual_letter"] = notch.map(ratings_mod.notch_to_sp_letter)
    # Kept only for transparency/audit (e.g. "how much does LOOCV differ from
    # the naive in-sample fit") -- nothing downstream reads these. Named
    # in_sample_* (not pred_notch) on purpose so nobody mistakes this for the
    # validated number.
    table["in_sample_pred_notch"] = in_sample_pred
    table["in_sample_pred_letter"] = in_sample_pred.round().map(ratings_mod.notch_to_sp_letter)
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
