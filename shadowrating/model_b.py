"""
Phase 3 / Model B: the statistical counterpart to Model A's fixed-weight
scorecard. Two estimators, both deliberately small and both evaluated only
out-of-sample (LOOCV) -- N~42 punishes anything more ambitious.

1. Ordered logit (mord.LogisticAT) on the 6-band rating scale
   (ratings.RATING_BANDS), not the full 21-notch scale -- fitting 20 cut
   points from 42 points isn't honest, fitting 5 is defensible. This is the
   headline model per CLAUDE.md: transparent, monotonic, lets the data choose
   pillar weights (vs. Model A's fixed ones).
2. Gradient boosting (XGBRegressor, shallow trees) on the continuous notch
   scale, for direct comparison against Model A's MAE/RMSE. Feature
   importances are reported as descriptive only -- with this little data they
   are not a reliable guide to what actually drives ratings.

Inputs to both are pillar scores (not raw indicators): 4-6 features for 42
countries is still a stretch, but 12+ raw indicators would guarantee
overfitting. Rows missing any feature or the target are dropped from that
estimator's training set, not imputed with a guess.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

from . import ratings as ratings_mod

# mord passes an unsupported scipy.optimize option through L-BFGS-B; harmless,
# but it fires once per LOOCV fold and floods stdout.
warnings.filterwarnings("ignore", message="Unknown solver options", category=OptimizeWarning)


def _complete_rows(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask = features.notna().all(axis=1) & target.notna()
    return features.loc[mask], target.loc[mask]


def fit_ordered_logit_loocv(pillar_scores: pd.DataFrame, notch: pd.Series) -> pd.DataFrame:
    """
    LOOCV predicted rating band (0..5) from an ordered logit fit on every
    other country. Returns a frame indexed like the inputs with columns
    band_true, band_pred (NaN for countries excluded from training).
    """
    import mord

    band = notch.map(ratings_mod.notch_to_band)
    X, y = _complete_rows(pillar_scores, band)

    out = pd.DataFrame(index=pillar_scores.index, columns=["band_true", "band_pred"], dtype=float)
    out["band_true"] = band

    for held_out in X.index:
        train_idx = X.index.drop(held_out)
        model = mord.LogisticAT()
        model.fit(X.loc[train_idx].to_numpy(), y.loc[train_idx].to_numpy())
        pred = model.predict(X.loc[[held_out]].to_numpy())[0]
        out.loc[held_out, "band_pred"] = pred

    return out


def fit_gbm_loocv(pillar_scores: pd.DataFrame, notch: pd.Series,
                  max_depth: int = 2, n_estimators: int = 40) -> tuple[pd.Series, pd.Series]:
    """
    LOOCV predicted notch from shallow gradient-boosted trees fit on every
    other country. Returns (predictions, mean_feature_importance) -- the
    latter is the average of each LOOCV fold's importances, reported as
    descriptive only (CLAUDE.md: "treat XGBoost feature importances as
    descriptive, not predictive" given N~42).
    """
    from xgboost import XGBRegressor

    X, y = _complete_rows(pillar_scores, notch)
    preds = pd.Series(np.nan, index=pillar_scores.index, dtype=float)
    importance_rows = []

    for held_out in X.index:
        train_idx = X.index.drop(held_out)
        model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators,
                              learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                              random_state=0)
        model.fit(X.loc[train_idx], y.loc[train_idx])
        preds.loc[held_out] = float(model.predict(X.loc[[held_out]])[0])
        importance_rows.append(pd.Series(model.feature_importances_, index=X.columns))

    mean_importance = pd.concat(importance_rows, axis=1).mean(axis=1).sort_values(ascending=False)
    return preds, mean_importance


def band_error_summary(band_table: pd.DataFrame) -> dict:
    valid = band_table.dropna(subset=["band_true", "band_pred"])
    diff = (valid["band_pred"].round() - valid["band_true"]).abs()
    return {
        "n": len(valid),
        "exact_match_rate": float((diff == 0).mean()),
        "within_one_band_rate": float((diff <= 1).mean()),
        "mae_bands": float(diff.mean()),
    }


def notch_error_summary(notch_true: pd.Series, notch_pred: pd.Series) -> dict:
    err = (notch_pred - notch_true).dropna()
    return {
        "n": len(err),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()),
    }
