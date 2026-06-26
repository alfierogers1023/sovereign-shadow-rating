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

Both LOOCV loops take the raw panel (not a precomputed pillar-score table)
and use `features.loocv_folds` so percentile-rank scaling is rebuilt on the
41 training countries every fold -- scaling the held-out country's features
against a distribution that includes itself would leak feature-engineering
information into a number reported as out-of-sample. See CLAUDE.md's
validation-integrity note and `scorecard.py`'s matching fix.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning

from . import features as features_mod
from . import ratings as ratings_mod

# mord passes an unsupported scipy.optimize option through L-BFGS-B; harmless,
# but it fires once per LOOCV fold and floods stdout.
warnings.filterwarnings("ignore", message="Unknown solver options", category=OptimizeWarning)


N_BANDS = len(ratings_mod.RATING_BANDS)
CI_MASS = 0.90  # central probability mass captured by the reported interval


def _complete_rows(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask = features.notna().all(axis=1) & target.notna()
    return features.loc[mask], target.loc[mask]


def _full_proba(model, X: np.ndarray) -> np.ndarray:
    """
    mord's `classes_` only lists bands actually seen in that fold's training
    set -- with N=42 and 6 bands this is normally all of them, but a fold
    could in principle drop the rarest band entirely. Reindex onto the full
    0..N_BANDS-1 range so every fold's probability vector is comparable and
    a never-seen band just gets 0 probability rather than silently shifting
    every other band's column index.
    """
    proba = model.predict_proba(X)
    full = np.zeros((X.shape[0], N_BANDS))
    for col, cls in enumerate(model.classes_):
        full[:, int(cls)] = proba[:, col]
    return full


def _central_interval(proba_row: np.ndarray, mass: float = CI_MASS) -> tuple[int, int]:
    """
    Smallest contiguous band range [lo, hi] whose cumulative probability
    covers at least `mass` -- the discrete analogue of a 90% interval.
    Rounds outward (>=, not >), which is the conservative direction: the
    reported interval never claims more confidence than the model actually
    has.
    """
    cdf = np.cumsum(proba_row)
    tail = (1 - mass) / 2
    lo = int(np.searchsorted(cdf, tail, side="left"))
    hi = int(np.searchsorted(cdf, 1 - tail, side="left"))
    return min(lo, len(proba_row) - 1), min(hi, len(proba_row) - 1)


def fit_ordered_logit_loocv(panel: pd.DataFrame, notch: pd.Series) -> pd.DataFrame:
    """
    LOOCV predicted rating band (0..5) from an ordered logit fit on every
    other country, with pillar scores rebuilt fold-wise from `panel` (see
    module docstring). Returns a frame indexed like `panel` with columns:
      band_true, band_pred       -- point estimate (kept, per the brief)
      band_ci_lower, band_ci_upper -- smallest contiguous band range
        covering >=90% of the fold's predicted probability mass
      band_proba_<i>             -- the full 6-band probability vector,
        for anyone who wants more than the interval (e.g. the dashboard)
      outside_ci                 -- True if the actual band falls outside
        [band_ci_lower, band_ci_upper] -- a stricter, statistically-aware
        divergence flag than "the point estimate differs," per CLAUDE.md
    NaN for countries excluded from training.
    """
    import mord

    band = notch.map(ratings_mod.notch_to_band)
    proba_cols = [f"band_proba_{i}" for i in range(N_BANDS)]
    cols = ["band_true", "band_pred", "band_ci_lower", "band_ci_upper"] + proba_cols
    out = pd.DataFrame(index=panel.index, columns=cols, dtype=float)
    out["band_true"] = band

    for held_out, train_pillars, _, held_pillars, _ in features_mod.loocv_folds(panel):
        if held_pillars.isna().any() or pd.isna(band.get(held_out)):
            continue
        y_train = band.reindex(train_pillars.index)
        X_train, y_train = _complete_rows(train_pillars, y_train)
        if len(X_train) < 2:
            continue
        model = mord.LogisticAT()
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        x_held = held_pillars.to_numpy().reshape(1, -1)
        pred = model.predict(x_held)[0]
        proba = _full_proba(model, x_held)[0]
        lo, hi = _central_interval(proba)

        out.loc[held_out, "band_pred"] = pred
        out.loc[held_out, "band_ci_lower"] = lo
        out.loc[held_out, "band_ci_upper"] = hi
        out.loc[held_out, proba_cols] = proba

    outside = (
        (out["band_true"] < out["band_ci_lower"]) | (out["band_true"] > out["band_ci_upper"])
    ).astype("boolean")
    outside[out["band_pred"].isna()] = pd.NA
    out["outside_ci"] = outside
    return out


def fit_gbm_loocv(panel: pd.DataFrame, notch: pd.Series,
                  max_depth: int = 2, n_estimators: int = 40) -> tuple[pd.Series, pd.Series]:
    """
    LOOCV predicted notch from shallow gradient-boosted trees fit on every
    other country, with pillar scores rebuilt fold-wise from `panel` (see
    module docstring). Returns (predictions, mean_feature_importance) -- the
    latter is the average of each LOOCV fold's importances, reported as
    descriptive only (CLAUDE.md: "treat XGBoost feature importances as
    descriptive, not predictive" given N~42).
    """
    from xgboost import XGBRegressor

    preds = pd.Series(np.nan, index=panel.index, dtype=float)
    importance_rows = []

    for held_out, train_pillars, _, held_pillars, _ in features_mod.loocv_folds(panel):
        if held_pillars.isna().any() or pd.isna(notch.get(held_out)):
            continue
        y_train = notch.reindex(train_pillars.index)
        X_train, y_train = _complete_rows(train_pillars, y_train)
        if len(X_train) < 2:
            continue
        model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators,
                              learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                              random_state=0)
        model.fit(X_train, y_train)
        preds.loc[held_out] = float(model.predict(held_pillars.to_frame().T)[0])
        importance_rows.append(pd.Series(model.feature_importances_, index=X_train.columns))

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
