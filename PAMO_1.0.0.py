"""PAMO 1.0.2 — on-demand nested accuracy audits, based on PAMO 1.0.1."""
from pathlib import Path
from io import BytesIO
from itertools import product, islice
import math
import os
import time
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Optional

VERSION = "1.0.2"
DEFAULT_RIDGE_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_RF_TREES_CV = 120
DEFAULT_RF_MIN_SAMPLES_LEAF = 2
SIMPLE_MODEL_TIE_TOL = 0.02
HYBRID_MIN_GAIN_TOL = 0.01
DEFAULT_RF_GRID = [dict(min_samples_leaf=leaf, max_depth=depth, max_features="sqrt")
                   for depth in (None, 12) for leaf in (1, 2, 4)]
SCORE_COLORS = [[0, "#b84040"], [0.5, "#efcf69"], [1, "#16804a"]]


def compute_df_signature(df) -> str:
    """Return a stable hash for a pandas DataFrame content + columns.

    This is used to make Streamlit caching invalidate correctly when the uploaded
    training data changes.
    """
    try:
        import pandas as _pd
        import numpy as _np
        h = _pd.util.hash_pandas_object(df, index=True).values
        # also include columns + dtypes
        meta = (tuple(map(str, df.columns)), tuple(map(str, df.dtypes)))
        meta_bytes = str(meta).encode('utf-8', errors='ignore')
        import hashlib
        m = hashlib.sha256()
        m.update(h.tobytes())
        m.update(meta_bytes)
        return m.hexdigest()
    except Exception:
        try:
            # Fallback: shape + columns only
            return f"fallback:{df.shape}:{tuple(map(str, df.columns))}"
        except Exception:
            return 'fallback:unknown'


def build_poly2_ridge(alpha: float) -> Pipeline:
    """Create a Poly(deg=2)+Scaler+Ridge pipeline that predicts a single objective."""
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha)))
    ])


def build_poly2_ridgecv(alphas) -> Pipeline:
    """Create a Poly(deg=2)+Scaler+RidgeCV(alphas) pipeline (single objective)."""
    alphas = [float(a) for a in alphas]
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        # cv=None -> efficient generalized CV inside each training fold (no leakage)
        ("ridge", RidgeCV(alphas=alphas, cv=None))
    ])


class RidgeSurrogate:
    """Wrapper to provide predict() + predict_uncertainty() with constant sigma_floor."""

    def __init__(self, model: Pipeline, sigma_floor: float = 0.0, chosen_alpha: Optional[float] = None):
        self.model = model
        self.sigma_floor = float(sigma_floor)
        self.chosen_alpha = float(chosen_alpha) if chosen_alpha is not None else None
        self.kind = "Ridge"

    def predict(self, X):
        return np.asarray(self.model.predict(np.asarray(X)))

    def predict_uncertainty(self, X):
        n = len(np.asarray(X))
        return np.full(n, self.sigma_floor, dtype=float)


class RandomForestSurrogate:
    """Wrapper to provide predict() + predict_uncertainty() for RF with tree-std + sigma_floor."""

    def __init__(self, model: RandomForestRegressor, sigma_floor: float = 0.0):
        self.model = model
        self.sigma_floor = float(sigma_floor)
        self.kind = "Random Forest"

    def predict(self, X):
        return np.asarray(self.model.predict(np.asarray(X)))

    def predict_uncertainty(self, X):
        X_arr = np.asarray(X)
        rf_std = 0.0
        if hasattr(self.model, "estimators_"):
            try:
                tree_preds = np.vstack([est.predict(X_arr) for est in self.model.estimators_])
                rf_std = np.std(tree_preds, axis=0)
            except Exception:
                rf_std = 0.0
        return np.sqrt(self.sigma_floor ** 2 + np.asarray(rf_std) ** 2)


class HybridEstimator(RegressorMixin, BaseEstimator):
    """sklearn-compatible estimator for the Hybrid surrogate.

    Hybrid model: y = Ridge(Poly2(x)) + RF_residual(x)
    """

    def __init__(
        self,
        ridge_alpha: float = 0.1,
        ridge_alphas: tuple = (0.001, 0.01, 0.1, 1.0, 10.0),
        ridge_auto: bool = True,
        rf_n_estimators: int = 150,
        rf_min_samples_leaf: int = 2,
        rf_max_depth=None,
        rf_max_features: str = "sqrt",
        n_jobs=None,
    ):
        # Store parameters exactly as passed for sklearn.clone compatibility
        self.ridge_alpha = ridge_alpha
        self.ridge_alphas = ridge_alphas
        self.ridge_auto = ridge_auto
        self.rf_n_estimators = rf_n_estimators
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_max_depth = rf_max_depth
        self.rf_max_features = rf_max_features
        self.n_jobs = n_jobs

        self._ridge_model = None
        self._rf_model = None

    def fit(self, X, y):
        import numpy as np
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)

        if self.ridge_auto:
            self._ridge_model = build_poly2_ridgecv(self.ridge_alphas)
        else:
            self._ridge_model = build_poly2_ridge(self.ridge_alpha)

        self._ridge_model.fit(X, y)
        resid = y - self._ridge_model.predict(X)

        n_jobs = None if self.n_jobs is None else int(self.n_jobs)

        self._rf_model = RandomForestRegressor(
            n_estimators=int(self.rf_n_estimators),
            random_state=42,
            n_jobs=n_jobs,
            max_features=self.rf_max_features,
            max_depth=self.rf_max_depth,
            min_samples_leaf=int(self.rf_min_samples_leaf),
        )
        self._rf_model.fit(X, resid)
        return self

    def predict(self, X):
        import numpy as np
        X = np.asarray(X)
        ridge_pred = self._ridge_model.predict(X)
        rf_pred = self._rf_model.predict(X)
        return ridge_pred + rf_pred


class HybridSurrogate:

    """
    Hybrid surrogate model:
      y_hat(x) = ridge_poly2(x) + rf_residual(x)

    Uncertainty:
      - sigma_floor: constant per objective (estimated from CV residuals of the hybrid model)
      - rf_std(x): std across residual RF trees (local epistemic proxy)
      - sigma_total(x) = sqrt(sigma_floor^2 + rf_std(x)^2)
    """

    def __init__(self, ridge_model: Pipeline, rf_residual_model: RandomForestRegressor, sigma_floor: float = 0.0):
        self.ridge_model = ridge_model
        self.rf_residual_model = rf_residual_model
        self.sigma_floor = float(sigma_floor)

    def predict(self, X):
        X_arr = np.asarray(X)
        base = np.asarray(self.ridge_model.predict(X_arr))
        if self.rf_residual_model is None:
            return base
        corr = np.asarray(self.rf_residual_model.predict(X_arr))
        return base + corr

    def predict_uncertainty(self, X):
        X_arr = np.asarray(X)
        rf_std = 0.0
        if self.rf_residual_model is not None and hasattr(self.rf_residual_model, "estimators_"):
            try:
                tree_preds = np.vstack([est.predict(X_arr) for est in self.rf_residual_model.estimators_])
                rf_std = np.std(tree_preds, axis=0)
            except Exception:
                rf_std = 0.0
        return np.sqrt(self.sigma_floor ** 2 + np.asarray(rf_std) ** 2)


def compute_param_sensitivity_from_poly_ridge(model: Pipeline, base_feature_names) -> np.ndarray:
    """
    Compute a parameter-level sensitivity vector (length = number of original parameters),
    derived from absolute Ridge coefficients in the polynomial feature space.

    This keeps the existing UI intact (sensitivity plots remain at parameter level).
    """
    try:
        poly = model.named_steps["poly"]
        ridge = model.named_steps["ridge"]

        # Names of expanded polynomial features (e.g., par1, par1 par2, par1^2)
        feature_names = poly.get_feature_names_out(input_features=np.array(base_feature_names, dtype=str))

        coefs = getattr(ridge, "coef_", None)
        if coefs is None:
            return np.zeros(len(base_feature_names), dtype=float)

        # Ridge with a single target -> coef_ shape (n_features,)
        coefs = np.asarray(coefs).reshape(-1)
        abs_coefs = np.abs(coefs)

        sensitivity = np.zeros(len(base_feature_names), dtype=float)
        base_feature_names = list(map(str, base_feature_names))

        for fname, w in zip(feature_names, abs_coefs):
            # Parse involved base features:
            # - "par1 par4" -> ["par1", "par4"]
            # - "par1^2"    -> ["par1"]
            involved = []
            for token in fname.split():
                if "^" in token:
                    token = token.split("^")[0]
                involved.append(token)

            for j, base in enumerate(base_feature_names):
                if base in involved:
                    sensitivity[j] += float(w)

        return sensitivity

    except Exception:
        return np.zeros(len(base_feature_names), dtype=float)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Return MAE, NMAE (MAE normalized by range), and R²."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mae = float(mean_absolute_error(y_true, y_pred))
    rng = float(np.ptp(y_true))
    nmae = float(mae / (rng + 1e-9))

    # R² is undefined if y is constant
    r2 = float(r2_score(y_true, y_pred)) if np.std(y_true) > 1e-12 else float('nan')
    return mae, nmae, r2


def _cv_eval_ridge(X: pd.DataFrame, y: np.ndarray, cv: KFold, ridge_mode: str, ridge_alpha: float,
                   ridge_alphas=DEFAULT_RIDGE_ALPHAS):
    y_pred = np.zeros(len(X), dtype=float)
    fold_scores = []
    fold_alphas = []

    for tr, va in cv.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        if ridge_mode == "auto":
            model = build_poly2_ridgecv(ridge_alphas)
        else:
            model = build_poly2_ridge(ridge_alpha)

        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        y_pred[va] = pred

        try:
            fold_scores.append(float(r2_score(y_va, pred)) if np.std(y_va) > 1e-12 else float('nan'))
        except Exception:
            fold_scores.append(float('nan'))

        # Alpha used (if RidgeCV)
        try:
            ridge_step = model.named_steps.get("ridge", None)
            if hasattr(ridge_step, "alpha_"):
                fold_alphas.append(float(ridge_step.alpha_))
        except Exception:
            pass

    mae, nmae, r2 = _compute_metrics(y, y_pred)
    sigma_floor = float(np.std(y - y_pred)) if len(y) > 1 else 0.0

    return {
        "name": "Ridge",
        "y_pred_cv": y_pred,
        "fold_scores": fold_scores,
        "mae": mae,
        "nmae": nmae,
        "r2": r2,
        "sigma_floor": sigma_floor,
        "fold_alphas": fold_alphas
    }


def _cv_eval_rf(X: pd.DataFrame, y: np.ndarray, cv: KFold, n_estimators: int, n_jobs: int,
                min_samples_leaf: int = DEFAULT_RF_MIN_SAMPLES_LEAF, max_depth=None, max_features="sqrt"):
    """Cross-validated out-of-fold predictions for Random Forest."""
    y_pred = np.zeros(len(X), dtype=float)
    fold_scores = []

    for tr, va in cv.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=42,
            n_jobs=int(n_jobs) if n_jobs is not None else None,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        y_pred[va] = pred

        try:
            fold_scores.append(float(r2_score(y_va, pred)) if np.std(y_va) > 1e-12 else float('nan'))
        except Exception:
            fold_scores.append(float('nan'))

    mae, nmae, r2 = _compute_metrics(y, y_pred)
    sigma_floor = float(np.std(y - y_pred)) if len(y) > 1 else 0.0

    return {
        "name": "Random Forest",
        "y_pred_cv": y_pred,
        "fold_scores": fold_scores,
        "mae": mae,
        "nmae": nmae,
        "r2": r2,
        "sigma_floor": sigma_floor
    }


def _cv_eval_hybrid(X: pd.DataFrame, y: np.ndarray, cv: KFold, ridge_mode: str, ridge_alpha: float,
                    n_estimators: int, n_jobs: int, ridge_alphas=DEFAULT_RIDGE_ALPHAS,
                    min_samples_leaf: int = DEFAULT_RF_MIN_SAMPLES_LEAF, max_depth=None, max_features="sqrt"):
    y_pred = np.zeros(len(X), dtype=float)
    fold_scores = []
    fold_alphas = []

    for tr, va in cv.split(X):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y[tr], y[va]

        # Ridge baseline
        if ridge_mode == "auto":
            ridge_model = build_poly2_ridgecv(ridge_alphas)
        else:
            ridge_model = build_poly2_ridge(ridge_alpha)

        ridge_model.fit(X_tr, y_tr)

        # Residual RF corrector
        resid_tr = y_tr - ridge_model.predict(X_tr)

        rf_model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=42,
            n_jobs=int(n_jobs) if n_jobs is not None else None,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf)
        )
        rf_model.fit(X_tr, resid_tr)

        pred = ridge_model.predict(X_va) + rf_model.predict(X_va)
        y_pred[va] = pred

        try:
            fold_scores.append(float(r2_score(y_va, pred)) if np.std(y_va) > 1e-12 else float('nan'))
        except Exception:
            fold_scores.append(float('nan'))

        # Alpha used (if RidgeCV)
        try:
            ridge_step = ridge_model.named_steps.get("ridge", None)
            if hasattr(ridge_step, "alpha_"):
                fold_alphas.append(float(ridge_step.alpha_))
        except Exception:
            pass

    mae, nmae, r2 = _compute_metrics(y, y_pred)
    sigma_floor = float(np.std(y - y_pred)) if len(y) > 1 else 0.0

    return {
        "name": "Hybrid (Ridge + RF residual)",
        "y_pred_cv": y_pred,
        "fold_scores": fold_scores,
        "mae": mae,
        "nmae": nmae,
        "r2": r2,
        "sigma_floor": sigma_floor,
        "fold_alphas": fold_alphas
    }


def _choose_model_kind(ridge_info, rf_info, hybrid_info):
    """
    Choose the model kind using NMAE with a simplicity bias:
      - If Ridge is within SIMPLE_MODEL_TIE_TOL of the best NMAE -> choose Ridge.
      - Choose Hybrid only if it improves over the best single model by > HYBRID_MIN_GAIN_TOL.
    """
    infos = {
        "Ridge": ridge_info,
        "Random Forest": rf_info,
        "Hybrid": hybrid_info
    }

    best_kind = min(infos.keys(), key=lambda k: infos[k]["nmae"])
    best_nmae = infos[best_kind]["nmae"]

    # Prefer Ridge if close to best
    if ridge_info["nmae"] <= best_nmae * (1.0 + SIMPLE_MODEL_TIE_TOL):
        return "Ridge"

    # If best is Hybrid, require meaningful gain over best single model
    if best_kind == "Hybrid":
        best_single = "Ridge" if ridge_info["nmae"] <= rf_info["nmae"] else "Random Forest"
        if infos[best_single]["nmae"] <= best_nmae * (1.0 + HYBRID_MIN_GAIN_TOL):
            # If Ridge is close to that best single model, still prefer Ridge
            if ridge_info["nmae"] <= infos[best_single]["nmae"] * (1.0 + SIMPLE_MODEL_TIE_TOL):
                return "Ridge"
            return best_single
        return "Hybrid"

    return best_kind


def _cv_select_best_rf(X: pd.DataFrame, y: np.ndarray, cv: KFold, n_estimators: int, n_jobs: int,
                       grid=DEFAULT_RF_GRID):
    """Evaluate a small RF grid and return the best config by CV NMAE."""
    best = None
    for params in grid:
        info = _cv_eval_rf(
            X, y, cv,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_depth=params["max_depth"],
            max_features=params["max_features"],
        )
        info["params"] = params
        if best is None or info["nmae"] < best["nmae"]:
            best = info
    return best


def _cv_select_best_hybrid(X: pd.DataFrame, y: np.ndarray, cv: KFold, ridge_mode: str, ridge_alpha: float,
                           n_estimators: int, n_jobs: int, grid=DEFAULT_RF_GRID):
    """Evaluate a small Hybrid grid (RF residual parameters) and return the best config by CV NMAE."""
    best = None
    for params in grid:
        info = _cv_eval_hybrid(
            X, y, cv,
            ridge_mode=ridge_mode,
            ridge_alpha=ridge_alpha,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_depth=params["max_depth"],
            max_features=params["max_features"],
        )
        info["params"] = params
        if best is None or info["nmae"] < best["nmae"]:
            best = info
    return best


def train_models(df, num_objectives, ridge_alpha=0.1, cv_folds=5,
                 model_selection_mode="Auto", objective_model_overrides=None,
                 rf_n_estimators=50, num_cores=1):
    validate_training_data(df, num_objectives)
    if objective_model_overrides is None:
        objective_model_overrides = {}

    X = df.drop(df.columns[-num_objectives:], axis=1)
    y_list = [df[col] for col in df.columns[-num_objectives:]]
    objective_names = list(df.columns[-num_objectives:])

    # CV setup
    n_splits = int(min(max(2, cv_folds), len(X)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Resources
    n_jobs = int(num_cores if num_cores is not None else (os.cpu_count() or 1))
    n_jobs = max(1, min(n_jobs, os.cpu_count() or n_jobs))

    rf_trees_final = int(rf_n_estimators)
    rf_trees_cv = int(min(DEFAULT_RF_TREES_CV, rf_trees_final))

    models = []
    plot_estimators = []  # sklearn estimators for learning curves
    cv_scores = []
    feature_importances = []
    objective_uncertainties = []
    chosen_model_kinds = []
    per_objective_reports = []

    for i, (obj_name, y_series) in enumerate(zip(objective_names, y_list)):
        y = np.asarray(y_series, dtype=float).reshape(-1)

        # Determine whether Ridge alpha is auto-tuned for this objective
        # Auto mode -> ridge_mode="auto"; Manual model selection -> ridge_mode="manual"
        ridge_mode = "auto" if str(model_selection_mode).lower().startswith("auto") else "manual"

        # Override per objective
        override = str(objective_model_overrides.get(obj_name, "Auto"))
        override_norm = override.strip().lower()

        # Evaluate candidates (needed for Auto or for "Auto" override inside Manual mode)
        ridge_info = _cv_eval_ridge(X, y, cv, ridge_mode=ridge_mode, ridge_alpha=float(ridge_alpha))
        rf_info = _cv_select_best_rf(X, y, cv, n_estimators=rf_trees_cv, n_jobs=n_jobs)
        hybrid_info = _cv_select_best_hybrid(X, y, cv, ridge_mode=ridge_mode, ridge_alpha=float(ridge_alpha), n_estimators=rf_trees_cv, n_jobs=n_jobs)

        # Decide model kind
        if override_norm in ("ridge", "random forest", "random_forest", "rf", "hybrid"):
            if override_norm == "ridge":
                chosen = "Ridge"
            elif override_norm in ("random forest", "random_forest", "rf"):
                chosen = "Random Forest"
            else:
                chosen = "Hybrid"
            selection_reason = f"Manual override: {override}"
        else:
            chosen = _choose_model_kind(ridge_info, rf_info, hybrid_info)
            selection_reason = "Auto selection (CV NMAE + simplicity bias)"

        chosen_model_kinds.append(chosen)

        # Select corresponding CV report
        chosen_info = {"Ridge": ridge_info, "Random Forest": rf_info, "Hybrid": hybrid_info}[chosen]
        sigma_floor = float(chosen_info["sigma_floor"])
        objective_uncertainties.append(sigma_floor)
        cv_scores.append(chosen_info["fold_scores"])

        # Train final model(s) on all data
        if chosen == "Ridge":
            if ridge_mode == "auto" and override_norm not in ("ridge",):
                final_ridge = build_poly2_ridgecv(DEFAULT_RIDGE_ALPHAS)
            else:
                final_ridge = build_poly2_ridge(float(ridge_alpha))
            final_ridge.fit(X, y)

            chosen_alpha = None
            try:
                ridge_step = final_ridge.named_steps.get("ridge", None)
                if hasattr(ridge_step, "alpha_"):
                    chosen_alpha = float(ridge_step.alpha_)
            except Exception:
                pass

            surrogate = RidgeSurrogate(final_ridge, sigma_floor=sigma_floor, chosen_alpha=chosen_alpha)
            models.append(surrogate)
            plot_estimators.append(final_ridge)

            # Sensitivity from polynomial ridge coefficients
            sens = compute_param_sensitivity_from_poly_ridge(final_ridge, X.columns)
            feature_importances.append(sens)

        elif chosen == "Random Forest":
            rf_params = (rf_info.get('params') or {})
            final_rf = RandomForestRegressor(
                n_estimators=int(rf_trees_final),
                random_state=42,
                n_jobs=int(n_jobs) if n_jobs is not None else None,
                max_features=rf_params.get('max_features', 'sqrt'),
                max_depth=rf_params.get('max_depth', None),
                min_samples_leaf=int(rf_params.get('min_samples_leaf', DEFAULT_RF_MIN_SAMPLES_LEAF)),
            )
            final_rf.fit(X, y)

            surrogate = RandomForestSurrogate(final_rf, sigma_floor=sigma_floor)
            models.append(surrogate)
            plot_estimators.append(final_rf)

            # Parameter sensitivity from RF feature_importances_ (already parameter-level)
            try:
                feature_importances.append(np.asarray(final_rf.feature_importances_, dtype=float))
            except Exception:
                feature_importances.append(np.zeros(len(X.columns), dtype=float))

        else:  # Hybrid
            if ridge_mode == "auto" and override_norm not in ("hybrid",):
                final_ridge = build_poly2_ridgecv(DEFAULT_RIDGE_ALPHAS)
                ridge_auto = True
            else:
                final_ridge = build_poly2_ridge(float(ridge_alpha))
                ridge_auto = False

            final_ridge.fit(X, y)
            resid_all = y - final_ridge.predict(X)

            hyb_params = (hybrid_info.get('params') or {})
            final_rf = RandomForestRegressor(
                n_estimators=int(rf_trees_final),
                random_state=42,
                n_jobs=int(n_jobs) if n_jobs is not None else None,
                max_features=hyb_params.get('max_features', 'sqrt'),
                max_depth=hyb_params.get('max_depth', None),
                min_samples_leaf=int(hyb_params.get('min_samples_leaf', DEFAULT_RF_MIN_SAMPLES_LEAF)),
            )
            final_rf.fit(X, resid_all)

            surrogate = HybridSurrogate(final_ridge, final_rf, sigma_floor=sigma_floor)
            surrogate.kind = "Hybrid"
            models.append(surrogate)

            # Provide a sklearn estimator for learning curves
            plot_estimators.append(HybridEstimator(
                ridge_alpha=float(ridge_alpha),
                ridge_alphas=DEFAULT_RIDGE_ALPHAS,
                ridge_auto=ridge_auto,
                rf_n_estimators=int(rf_trees_final),
                rf_min_samples_leaf=int(hyb_params.get('min_samples_leaf', DEFAULT_RF_MIN_SAMPLES_LEAF)),
                rf_max_depth=hyb_params.get('max_depth', None),
                rf_max_features=hyb_params.get('max_features', 'sqrt'),
                n_jobs=int(n_jobs) if n_jobs is not None else None
            ))

            sens = compute_param_sensitivity_from_poly_ridge(final_ridge, X.columns)
            feature_importances.append(sens)

        # Build report for UI
        per_objective_reports.append({
            "Objective": obj_name,
            "Selected model": chosen,
            "Selection basis": selection_reason,
            "Ridge CV NMAE": ridge_info["nmae"],
            "RF CV NMAE": rf_info["nmae"],
            "Hybrid CV NMAE": hybrid_info["nmae"],
            "Ridge CV R2": ridge_info["r2"],
            "RF CV R2": rf_info["r2"],
            "Hybrid CV R2": hybrid_info["r2"],
            "Selected CV MAE": chosen_info["mae"],
            "Selected CV NMAE": chosen_info["nmae"],
            "Selected CV R2": chosen_info["r2"],
            "Uncertainty floor (CV resid std)": sigma_floor,
        })


    return dict(models=models, plot_estimators=plot_estimators, cv_scores=cv_scores,
                feature_importances=feature_importances, objective_uncertainties=objective_uncertainties,
                model_kinds=chosen_model_kinds, reports=per_objective_reports,
                param_names=list(X.columns), objective_names=objective_names,
                sample=X.iloc[0].to_numpy(), training_data=df.copy())



def validate_training_data(df, num_objectives):
    if not 1 <= int(num_objectives) < len(df.columns):
        raise ValueError('Keep at least one parameter column before the objective columns.')
    if len(df) < 4:
        raise ValueError('At least four complete training cases are required.')
    if not df.columns.is_unique:
        raise ValueError('Column names must be unique.')
    try:
        values = df.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('All parameters and objectives must be numeric.') from exc
    if not np.isfinite(values).all():
        raise ValueError('Remove or correct missing and non-finite training values.')


def training_signature(df, num_objectives, settings, overrides):
    return (VERSION, compute_df_signature(df), int(num_objectives),
            tuple(sorted(settings.items())), tuple(sorted(overrides.items())))


def get_training_bundle(df, num_objectives, settings, overrides, slot):
    """Session-local models: full data content and all training settings identify a fit."""
    signature = training_signature(df, num_objectives, settings, overrides)
    cached = st.session_state.get(slot)
    if cached is None or cached['signature'] != signature:
        bundle = train_models(df, num_objectives, objective_model_overrides=overrides, **settings)
        bundle['signature'] = signature
        bundle['learning_curves'] = None
        st.session_state[slot] = bundle
    return st.session_state[slot]


@st.cache_data(show_spinner=False, max_entries=8)
def read_table(content, filename):
    stream = BytesIO(content)
    return pd.read_csv(stream) if filename.lower().endswith('.csv') else pd.read_excel(stream)


def normalized_scores(values, weights, bounds=None):
    """Higher is better for every direction. Zero weights exclude an objective."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(weights):
        raise ValueError('Objective count and weight count must match.')
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError('Objectives and weights must be finite.')
    low, high = (values.min(axis=0), values.max(axis=0)) if bounds is None else bounds
    span = np.asarray(high) - np.asarray(low)
    scaled = np.divide(values - low, span, out=np.zeros_like(values), where=span > 0)
    utility = np.where(weights < 0, 1.0 - scaled, scaled)
    utility[:, span == 0] = 0.0  # Constant objectives cannot discriminate between solutions.
    return utility @ np.abs(weights)


def pareto_mask(values, weights):
    """Exact dominance, preserving ties. Weight magnitude affects score, not membership."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(weights):
        raise ValueError('Objective count and weight count must match.')
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError('Cannot compute Pareto membership from non-finite values.')
    active = weights != 0
    if not active.any():
        raise ValueError('Give at least one objective a nonzero weight.')
    utility = values[:, active] * np.sign(weights[active])
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        if keep[i]:
            # Current surviving points worse in all objectives are dominated.
            survivors = np.flatnonzero(keep)
            dominated = (utility[survivors] <= utility[i]).all(axis=1) & (utility[survivors] < utility[i]).any(axis=1)
            keep[survivors[dominated]] = False
    return keep


def parameter_grid(ranges):
    axes = []
    for name, (low, high, step) in ranges.items():
        if not np.isfinite([low, high, step]).all() or low > high:
            raise ValueError(f'{name}: use finite values and a minimum no greater than the maximum.')
        if low == high:
            axes.append(np.array([low]))
        else:
            if step <= 0:
                raise ValueError(f'{name}: step must be positive.')
            count = int(math.floor((high - low) / step + 1e-10)) + 1
            axis = low + np.arange(count) * step
            # Never generate a point beyond the entered maximum.
            axes.append(np.minimum(axis, high))
    return axes, math.prod(len(axis) for axis in axes)


def evaluate_grid(bundle, ranges, weights, progress=None):
    axes, total = parameter_grid(ranges)
    if not any(weights):
        raise ValueError('Give at least one objective a nonzero weight.')
    iterator = product(*axes)
    chunks = []
    completed = 0
    while batch := list(islice(iterator, 2048)):
        x = np.asarray(batch, dtype=float)
        predicted = np.column_stack([model.predict(x) for model in bundle['models']])
        uncertainty = np.column_stack([model.predict_uncertainty(x) for model in bundle['models']])
        if not np.isfinite(predicted).all() or not np.isfinite(uncertainty).all():
            raise ValueError('Model produced invalid predictions; optimization was stopped.')
        chunks.append(np.column_stack([x, predicted, uncertainty]))
        completed += len(batch)
        if progress:
            progress(completed / total)
    columns = bundle['param_names'] + bundle['objective_names'] + ['Uncertainty_' + o for o in bundle['objective_names']]
    results = pd.DataFrame(np.vstack(chunks), columns=columns)
    values = results[bundle['objective_names']].to_numpy()
    results['Weighted Sum'] = normalized_scores(values, weights)
    results['Pareto'] = pareto_mask(values, weights)
    return results


def signed_sensitivity(bundle):
    """Average model response across a feature's observed range (not causal importance)."""
    if 'signed_sensitivity' in bundle:
        return bundle['signed_sensitivity']
    data = bundle['training_data']
    names = bundle['param_names']
    reference = data[names]
    if len(reference) > 256:
        reference = reference.sample(256, random_state=42)
    x = reference.to_numpy(dtype=float)
    effects = np.zeros((len(bundle['models']), len(names)))
    changes = np.zeros_like(effects)
    mixed = np.zeros_like(effects, dtype=bool)
    for j, name in enumerate(names):
        low, high = float(data[name].min()), float(data[name].max())
        if low == high:
            continue
        blocks = []
        levels = 9
        for level in np.linspace(low, high, levels):
            block = x.copy()
            block[:, j] = level
            blocks.append(block)
        probe = np.vstack(blocks)
        for i, (model, objective) in enumerate(zip(bundle['models'], bundle['objective_names'])):
            predictions = np.asarray(model.predict(probe), dtype=float).reshape(levels, len(x))
            if not np.isfinite(predictions).all():
                raise ValueError(f'Invalid predictions while calculating sensitivity for {objective}.')
            response = predictions.mean(axis=1)
            delta = float(response[-1] - response[0])
            span = float(data[objective].max() - data[objective].min())
            changes[i, j] = delta
            effects[i, j] = 100 * delta / span if span > 0 else 0.0
            increments = np.diff(response)
            tolerance = max(span * 1e-6, 1e-10)
            mixed[i, j] = bool((increments > tolerance).any() and (increments < -tolerance).any())
    result = dict(values=effects, changes=changes, mixed=mixed, reference_count=len(x))
    bundle['signed_sensitivity'] = result
    return result


def sensitivity_figure(bundle):
    stats = signed_sensitivity(bundle)
    objectives = bundle['objective_names']
    fig = make_subplots(rows=1, cols=len(objectives), horizontal_spacing=min(0.2, 0.4 / len(objectives)),
                        subplot_titles=[f'{name} ({kind})' for name, kind in zip(objectives, bundle['model_kinds'])])
    for i, values in enumerate(stats['values']):
        order = np.argsort(values)
        limit = max(float(np.max(np.abs(values))), 1.0)
        labels = [bundle['param_names'][j] + (' *' if stats['mixed'][i, j] else '') for j in order]
        fig.add_trace(go.Bar(x=values[order], y=labels, orientation='h', showlegend=False,
            customdata=np.column_stack([stats['changes'][i, order], stats['mixed'][i, order]]),
            marker=dict(color=values[order], colorscale=SCORE_COLORS, cmin=-limit, cmax=limit,
                        line=dict(color='#777777', width=0.4)),
            hovertemplate='%{y}<br>Net sensitivity: %{x:.2f}% of objective range<br>Change in objective units: %{customdata[0]:.3f}<extra></extra>'),
            row=1, col=i + 1)
        fig.update_xaxes(range=[-limit * 1.15, limit * 1.15], zeroline=True,
                         zerolinewidth=2, zerolinecolor='#666666', title_text='Net sensitivity (% of objective range)', row=1, col=i + 1)
        fig.update_yaxes(automargin=True, row=1, col=i + 1)
    fig.update_layout(template='plotly_white', height=max(380, 30 * len(bundle['param_names']) + 150),
                      margin=dict(l=20, r=20, b=60, t=65))
    return fig


def render_sensitivity(bundle, key):
    st.subheader('Sensitivity Analysis')
    st.plotly_chart(sensitivity_figure(bundle), use_container_width=True, key=key)
    st.caption('Negative/red = inverse response; positive/green = direct response; zero/yellow = no net change. '
               'Bars show the mean predicted change from each parameter’s observed minimum to maximum, '
               'as a percentage of the observed objective range. Colors indicate direction, not desirability.')
    st.caption('* Direction reverses along the sampled response curve; a small net bar can hide a strong nonlinear response. '
               'Other inputs are held at their reference-case values (up to 256 cases). Constant parameters have zero sensitivity. '
               'These model-based effects are not causal and can include unobserved combinations of correlated inputs.')


def accuracy_splits(data, params, mode, folds, repeats, parameter=None, groups=None):
    """Outer tests never share cases (or identical input combinations) with training."""
    if mode == 'Unseen parameter levels':
        labels = data[parameter].to_numpy()
        levels = np.unique(labels)
        if not 2 <= len(levels) <= 20:
            raise ValueError('Choose a parameter with 2–20 distinct levels.')
        splits = []
        for level in levels:
            test = np.flatnonzero(labels == level)
            kind = 'boundary extrapolation' if level in (levels[0], levels[-1]) else 'interior interpolation'
            splits.append((1, f'{parameter}={level:g} ({kind})', np.flatnonzero(labels != level), test))
    else:
        if mode == 'Unseen buildings' and groups is None:
            raise ValueError('Building labels are required for an unseen-building audit.')
        labels = (np.asarray(groups) if mode == 'Unseen buildings' else
                  pd.util.hash_pandas_object(data[params], index=False).to_numpy())
        if len(labels) != len(data) or pd.isna(labels).any():
            raise ValueError('Supply one nonempty building label for every training row, in the same order.')
        codes, unique = pd.factorize(labels)
        if len(unique) < folds:
            raise ValueError(f'Need at least {folds} distinct buildings or input combinations.')
        splits = []
        for repeat in range(1, repeats + 1):
            for fold, (tr, te) in enumerate(KFold(folds, shuffle=True, random_state=41 + repeat).split(unique), 1):
                splits.append((repeat, f'Fold {fold}', np.flatnonzero(np.isin(codes, tr)), np.flatnonzero(np.isin(codes, te))))
    if any(len(tr) < 8 or len(te) < 1 for _, _, tr, te in splits):
        raise ValueError('Every audit split needs at least 8 training cases and 1 test case. Add data or adjust the audit.')
    return splits


def run_accuracy_audit(data, num_objectives, settings, overrides, mode, folds=5,
                       repeats=1, parameter=None, groups=None, progress=None):
    params, objectives = list(data.columns[:-num_objectives]), list(data.columns[-num_objectives:])
    splits = accuracy_splits(data, params, mode, folds, repeats, parameter, groups)
    rows = []
    for step, (repeat, split, tr, te) in enumerate(splits, 1):
        training = data.iloc[tr].reset_index(drop=True)
        fitted = train_models(training, num_objectives, objective_model_overrides=overrides, **settings)
        x = data.iloc[te][params]
        for obj, model, kind in zip(objectives, fitted['models'], fitted['model_kinds']):
            predicted = np.asarray(model.predict(x), dtype=float)
            if not np.isfinite(predicted).all():
                raise ValueError(f'Non-finite audit predictions for {obj}.')
            for pos, index in enumerate(te):
                actual = float(data.iloc[index][obj])
                rows.append({'Repeat': repeat, 'Split': split, 'Data row': int(index) + 1,
                    'Objective': obj, 'Actual': actual, 'Prediction': float(predicted[pos]),
                    'Error': float(predicted[pos]) - actual,
                    'Absolute error': abs(float(predicted[pos]) - actual),
                    'Baseline prediction': float(training[obj].mean()), 'Selected model': kind,
                    **{f'Input: {p}': data.iloc[index][p] for p in params}})
        if progress:
            progress(step / len(splits), f'Completed audit split {step} of {len(splits)}')
    return pd.DataFrame(rows)


def audit_metrics(predictions, tolerances, by_split=False):
    records = []
    keys = ['Objective', 'Repeat'] + (['Split'] if by_split else [])
    for group, frame in predictions.groupby(keys, sort=False):
        y, pred = frame['Actual'].to_numpy(), frame['Prediction'].to_numpy()
        error = pred - y
        absolute = np.abs(error)
        obj = group[0]
        records.append({**dict(zip(keys, group)), 'Test cases': len(y),
            'MAE': absolute.mean(), 'RMSE': np.sqrt(np.mean(error ** 2)),
            'R²': r2_score(y, pred) if len(y) > 1 and np.ptp(y) > 0 else np.nan,
            'Bias': error.mean(), '90th percentile absolute error': np.quantile(absolute, .9),
            'Maximum absolute error': absolute.max(), 'Tolerance (±)': tolerances[obj],
            'Within tolerance (%)': 100 * np.mean(absolute <= tolerances[obj]),
            'Training-mean baseline MAE': np.mean(np.abs(y - frame['Baseline prediction'].to_numpy()))})
    return pd.DataFrame(records)


def render_accuracy_audit(data, num_objectives, settings, overrides, key):
    import json
    import zipfile
    params, objectives = list(data.columns[:-num_objectives]), list(data.columns[-num_objectives:])
    signature = training_signature(data, num_objectives, settings, overrides)
    st.caption('On-demand nested validation: each test case is excluded from training and model selection. '
               'Uses the current model settings; it does not replace your trained models. '
               'This measures agreement with the supplied data, not real-building accuracy.')
    mode = st.selectbox('Accuracy test', ['Held-out combinations', 'Unseen parameter levels', 'Unseen buildings'], key=key + '_mode')
    groups = None
    if mode == 'Unseen buildings':
        st.caption('Upload a one-column CSV with a header and one building ID per training row, in exactly the same order. '
                   'IDs are used only for splitting, never as model inputs. All rows of a building stay together in the outer test.')
        upload = st.file_uploader('Building labels CSV', type=['csv'], key=key + '_groups')
        if upload is not None:
            labels = pd.read_csv(upload)
            if labels.shape[1] == 1 and len(labels) == len(data) and not labels.isna().any().any():
                groups = labels.iloc[:, 0].astype(str).tolist()
            else:
                st.error('The labels file must have exactly one column and one nonempty label per training row.')
    with st.form(key + '_form'):
        parameter = st.selectbox('Parameter to withhold by level', params, key=key + '_param') if mode == 'Unseen parameter levels' else None
        folds = st.slider('Outer test folds', 3, 5, 5, key=key + '_folds') if mode != 'Unseen parameter levels' else 5
        repeats = st.slider('Repeated tests', 1, 3, 1, key=key + '_repeats') if mode != 'Unseen parameter levels' else 1
        tolerances = {o: st.number_input(f'Acceptable absolute error — {o}', min_value=0.0,
            value=float(np.ptp(data[o].to_numpy()) * .05), key=key + '_tol_' + str(i)) for i, o in enumerate(objectives)}
        st.caption('Tolerances use the original objective units. Defaults are 5% of each observed range, not acceptance recommendations. '
                   'For SDA stored on a 0–100 scale, enter 5 for ±5 percentage points.')
        st.caption('Each outer split retrains and selects models using only its training cases. '
                   'Repeated tests and parameter-level tests can take several minutes. '
                   'A grouped outer test holds buildings out entirely; internal model selection uses the standard row-based CV.')
        submitted = st.form_submit_button('Run accuracy audit')
        st.form_submit_button('Apply tolerances without retraining')
    config = dict(mode=mode, folds=folds, repeats=repeats, parameter=parameter, groups=groups)
    identity = (signature, json.dumps(config, sort_keys=True))
    slot = key + '_result'
    if submitted:
        if mode == 'Unseen buildings' and groups is None:
            st.error('Upload valid building labels before running this test.')
        else:
            bar = st.progress(0.0)
            try:
                with st.spinner('Training independent audit models; this can take several minutes...'):
                    predictions = run_accuracy_audit(data, num_objectives, settings, overrides, **config,
                        progress=lambda value, text: bar.progress(value, text=text))
                st.session_state[slot] = (identity, predictions)
            except Exception as exc:
                st.error(f'Accuracy audit could not complete: {exc}')
            finally:
                bar.empty()
    stored = st.session_state.get(slot)
    if stored is None or stored[0] != identity:
        st.info('Run the audit to see results for these data, model settings and test configuration.')
        return
    predictions = stored[1]
    st.caption('Positive error means overprediction. Repeats reuse the same cases; they are not additional independent data. '
               'R² is not percentage accuracy and is undefined for constant test targets. '
               'Identical inputs stay together in the combinations test. Input-level and building tests assess harder generalisation.')
    metrics = audit_metrics(predictions, tolerances)
    st.dataframe(metrics, hide_index=True, width='stretch')
    with st.expander('Results by test split'):
        st.dataframe(audit_metrics(predictions, tolerances, True), hide_index=True, width='stretch')
    objective = st.selectbox('Inspect objective', objectives, key=key + '_objective')
    repeat = st.selectbox('Inspect repeat', sorted(predictions['Repeat'].unique()), key=key + '_repeat')
    selected = predictions[(predictions.Objective == objective) & (predictions.Repeat == repeat)]
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Prediction versus actual', 'Prediction error'))
    fig.add_trace(go.Scatter(x=selected.Actual, y=selected.Prediction, mode='markers',
        customdata=selected[['Data row']], hovertemplate='Data row %{customdata[0]}<br>Actual: %{x}<br>Predicted: %{y}<extra></extra>', name='Held-out cases'), row=1, col=1)
    low, high = min(selected.Actual.min(), selected.Prediction.min()), max(selected.Actual.max(), selected.Prediction.max())
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode='lines', line=dict(dash='dash'), name='Perfect prediction'), row=1, col=1)
    fig.add_trace(go.Scatter(x=selected.Actual, y=selected.Error, mode='markers', name='Error'), row=1, col=2)
    fig.add_hline(y=0, row=1, col=2)
    for sign in (-1, 1):
        fig.add_hline(y=sign * tolerances[objective], line_dash='dot', line_color='#16804a', row=1, col=2)
    fig.update_xaxes(title_text='Actual supplied value')
    fig.update_yaxes(title_text='Prediction', row=1, col=1)
    fig.update_yaxes(title_text='Prediction minus actual', row=1, col=2)
    fig.update_layout(height=420, title=objective)
    st.plotly_chart(fig, width='stretch', key=key + '_charts')
    st.write('Largest prediction errors — selected objective and repeat')
    st.dataframe(selected.sort_values('Absolute error', ascending=False).head(20), hide_index=True, width='stretch')
    if st.button('Prepare audit download', key=key + '_prepare_audit'):
        archive = BytesIO()
        metadata = dict(version=VERSION, data_signature=compute_df_signature(data), settings=settings,
                        overrides=overrides, configuration=config, tolerances=tolerances,
                        note='Data row is 1-based within the imported table, excluding the header. Errors use original units.')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('predictions.csv', predictions.to_csv(index=False))
            z.writestr('metrics.csv', metrics.to_csv(index=False))
            z.writestr('metrics_by_split.csv', audit_metrics(predictions, tolerances, True).to_csv(index=False))
            z.writestr('configuration.json', json.dumps(metadata, indent=2, default=str))
        st.session_state[key + '_download'] = (identity, dict(tolerances), archive.getvalue())
    prepared = st.session_state.get(key + '_download')
    if prepared and prepared[:2] == (identity, tolerances):
        st.download_button('Download accuracy audit', prepared[2], 'pamo_accuracy_audit.zip',
            'application/zip', key=key + '_download_button', on_click='ignore')


def render_verification(bundle, key):
    st.dataframe(pd.DataFrame(bundle['reports']), hide_index=True, use_container_width=True)
    fig = go.Figure()
    for name, scores in zip(bundle['objective_names'], bundle['cv_scores']):
        fig.add_trace(go.Box(y=scores, name=name, boxpoints='all'))
    fig.update_layout(title='Cross-Validation Performance by Objective', yaxis_title='R²', height=350)
    st.plotly_chart(fig, use_container_width=True, key=key + '_cv')
    render_sensitivity(bundle, key + '_sensitivity')
    st.subheader('Testing Model Predictions')
    st.write(f"Sample input: {bundle['sample']}")
    for i, (name, model) in enumerate(zip(bundle['objective_names'], bundle['models'])):
        st.write(f'Objective {i + 1} ({name}) prediction: {model.predict([bundle["sample"]])[0]:.2f}')


def render_learning_curves(bundle, folds, key):
    if st.button('Calculate learning curves', key=key + '_calculate'):
        with st.spinner('Calculating learning curves...'):
            data = bundle['training_data']
            cv = KFold(n_splits=min(folds, len(data) // 2), shuffle=True, random_state=42)
            curves = []
            for name, estimator in zip(bundle['objective_names'], bundle['plot_estimators']):
                sizes, train, test = learning_curve(estimator, data[bundle['param_names']], data[name],
                    cv=cv, train_sizes=np.linspace(0.4, 1, 5), scoring='r2', n_jobs=1)
                curves.append((name, sizes, np.nanmean(train, axis=1), np.nanmean(test, axis=1),
                               np.nanstd(train, axis=1), np.nanstd(test, axis=1)))
            bundle['learning_curves'] = curves
    if bundle.get('learning_curves') is not None:
        st.caption('Shaded bands show ±1 standard deviation across validation folds. '
                   'They describe fold variability, not prediction confidence intervals.')
        for name, sizes, train, test, train_std, test_std in bundle['learning_curves']:
            fig = go.Figure()
            for label, mean, spread, color, fill in (
                ('Training R²', train, train_std, '#16804a', 'rgba(22,128,74,0.15)'),
                ('CV R²', test, test_std, '#326b9a', 'rgba(50,107,154,0.15)')):
                fig.add_trace(go.Scatter(x=sizes, y=mean - spread, mode='lines',
                    line=dict(width=0), showlegend=False, legendgroup=label, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=sizes, y=mean + spread, mode='lines',
                    line=dict(width=0), fill='tonexty', fillcolor=fill,
                    showlegend=False, legendgroup=label, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=sizes, y=mean, name=label, legendgroup=label,
                    mode='lines+markers', line_color=color, customdata=spread,
                    hovertemplate='Training samples: %{x}<br>Mean R²: %{y:.3f}'
                                  '<br>Fold SD: %{customdata:.3f}<extra>%{fullData.name}</extra>'))
            fig.update_layout(title=f'Learning Curve — {name}', xaxis_title='Training samples', yaxis_title='R²')
            st.plotly_chart(fig, use_container_width=True, key=key + '_' + name)


def parallel_figure(df, parameters, objectives, title):
    dimensions = []
    for name in parameters + objectives + ['Weighted Sum']:
        low, high = float(df[name].min()), float(df[name].max())
        if low == high:
            low, high = low - 0.1, high + 0.1
        dimensions.append(dict(label='Score' if name == 'Weighted Sum' else name,
                               values=df[name], range=[low, high]))
    fig = go.Figure(go.Parcoords(dimensions=dimensions, line=dict(
        color=df['Weighted Sum'], colorscale=SCORE_COLORS, showscale=True,
        colorbar=dict(title='Score')), unselected=dict(line=dict(opacity=0.064))))
    fig.update_layout(title=title, height=580, margin=dict(t=100, b=40, l=50, r=60))
    return fig


def trend_statistics(results, parameter, objective):
    """Descriptive summaries at exact observed parameter levels; no interpolation/binning."""
    grouped = results.groupby(parameter, sort=True)[objective]
    summary = grouped.agg(['mean', 'median', 'count', 'min', 'max'])
    summary['q25'] = grouped.quantile(0.25)
    summary['q75'] = grouped.quantile(0.75)
    return summary


def trend_figure(results, bundle, param):
    fig = make_subplots(rows=1, cols=len(bundle['objective_names']), subplot_titles=bundle['objective_names'])
    for col, obj in enumerate(bundle['objective_names'], 1):
        stats = trend_statistics(results, param, obj)
        details = np.column_stack([stats['count'], stats['q25'], stats['q75'], stats['min'], stats['max']])
        fig.add_trace(go.Scattergl(x=results[param], y=results[obj], mode='markers',
            marker=dict(size=3, opacity=0.18, color='#33966e'), name='All solutions',
            legendgroup='solutions', showlegend=col == 1), row=1, col=col)
        # A single observation has no observed within-level spread; omit its band.
        lower = stats['q25'].where(stats['count'] >= 2)
        upper = stats['q75'].where(stats['count'] >= 2)
        fig.add_trace(go.Scatter(x=stats.index, y=lower, mode='lines', line=dict(width=0),
            hoverinfo='skip', legendgroup='iqr', showlegend=False, connectgaps=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=stats.index, y=upper, mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(70,130,180,0.20)', name='Middle 50% (P25–P75)',
            legendgroup='iqr', showlegend=col == 1, connectgaps=False, hoverinfo='skip'), row=1, col=col)
        hover = ('Parameter: %{x}<br>%{fullData.name}: %{y:.3f}<br>Cases: %{customdata[0]:.0f}'
                 '<br>P25–P75: %{customdata[1]:.3f} – %{customdata[2]:.3f}'
                 '<br>Min–max: %{customdata[3]:.3f} – %{customdata[4]:.3f}<extra></extra>')
        fig.add_trace(go.Scatter(x=stats.index, y=stats['mean'], mode='lines+markers',
            line=dict(color='#d57532'), name='Mean', legendgroup='mean', showlegend=col == 1,
            customdata=details, hovertemplate=hover), row=1, col=col)
        fig.add_trace(go.Scatter(x=stats.index, y=stats['median'], mode='lines+markers',
            line=dict(color='#326b9a', dash='dash'), marker=dict(symbol='diamond', size=5),
            name='Median', legendgroup='median', showlegend=col == 1,
            customdata=details, hovertemplate=hover), row=1, col=col)
        fig.update_xaxes(title_text=param, row=1, col=col)
        fig.update_yaxes(title_text=obj, row=1, col=col)
    fig.update_layout(title=f'Parameter: {param}', height=390,
                      legend=dict(orientation='h', y=-0.3, x=0), margin=dict(b=100))
    return fig


def render_trends(results, bundle, key, source_label='evaluated combinations'):
    st.caption(f'Trends use all {len(results):,} {source_label}. Mean and median summarize each exact parameter value. '
               'The shaded band contains the middle 50% of solution values (25th–75th percentiles). '
               'Hover on a line for case counts and ranges; click legend entries to show or hide indicators.')
    st.caption('The band describes variation as other parameters change—not a confidence interval or model uncertainty. '
               'No band is shown for single-case levels. Lines connect observed levels only; they are not fitted predictions between levels. '
               'For a constant parameter there is only one summary point, so no directional trend can be inferred.')
    for param in bundle['param_names']:
        st.plotly_chart(trend_figure(results, bundle, param), use_container_width=True, key=f'{key}_{param}')
    if bundle.get('feature_importances') is not None:
        render_sensitivity(bundle, key + '_sensitivity')
    else:
        st.info('Estimate objectives with the current dataset and model settings to show sensitivity diagrams here.')


def excel_bytes(df):
    stream = BytesIO()
    df.to_excel(stream, index=False, engine='openpyxl')
    return stream.getvalue()


def prepare_table_export(df, filename):
    """Return bytes, actual filename, MIME type and optional fallback explanation."""
    target = Path(filename)
    if target.suffix.lower() not in ('.xlsx', '.csv'):
        raise ValueError('Use an .xlsx or .csv filename.')
    reason = None
    if target.suffix.lower() == '.xlsx':
        if len(df) > 1_048_575 or len(df.columns) > 16_384:
            reason = 'The table exceeds Excel worksheet limits.'
        else:
            try:
                return excel_bytes(df), str(target), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', None
            except Exception as exc:
                reason = f'Excel export failed: {exc}'
        target = target.with_suffix('.csv')
    return df.to_csv(index=False).encode('utf-8-sig'), str(target), 'text/csv', reason


def save_table_export(df, filename):
    payload, actual, _, reason = prepare_table_export(df, filename)
    try:
        Path(actual).write_bytes(payload)
    except OSError as exc:
        if Path(actual).suffix.lower() != '.xlsx':
            raise
        actual = str(Path(actual).with_suffix('.csv'))
        Path(actual).write_bytes(df.to_csv(index=False).encode('utf-8-sig'))
        reason = f'Excel file could not be written: {exc}'
    return actual, reason


def download_table(df, filename, key):
    # Generate a download only on request; filter interactions do not serialize Excel.
    signature = compute_df_signature(df)
    slot = key + '_prepared'
    if st.button('Prepare download (Excel / CSV fallback)', key=key + '_prepare'):
        try:
            st.session_state[slot] = (signature, prepare_table_export(df, filename))
        except Exception as exc:
            st.error(f'Could not prepare the download: {exc}. Results remain available.')
    cached = st.session_state.get(slot)
    if cached and cached[0] == signature:
        payload, actual, mime, reason = cached[1]
        if reason:
            st.warning(reason + ' CSV is available instead.')
        st.download_button('Download ' + Path(actual).suffix[1:].upper(), payload, Path(actual).name,
            mime, key=key + '_download', on_click='ignore')


def filter_results(df, parameters, objectives, key):
    with st.form(key + '_filter_form'):
        st.caption('Set ranges, then apply all filters together. Higher scores are better and appear at the top in green.')
        filters = {}
        for i, name in enumerate(parameters + objectives + ['Weighted Sum']):
            low, high = float(df[name].min()), float(df[name].max())
            label = 'Score' if name == 'Weighted Sum' else name
            if low == high:
                st.caption(f'{label}: fixed at {low:g}')
                filters[name] = (low, high)
            else:
                filters[name] = st.slider(label, low, high, (low, high), key=f'{key}_filter_{i}')
        submitted = st.form_submit_button('Apply filters')
    slot = key + '_filters'
    if submitted or slot not in st.session_state:
        st.session_state[slot] = filters
    filtered = df
    for name, (low, high) in st.session_state[slot].items():
        if name in filtered:
            filtered = filtered[filtered[name].between(low, high)]
    st.write(f'Filtered to {len(filtered):,} solutions from {len(df):,}.')
    if filtered.empty:
        st.info('No solutions match these ranges.')
    else:
        st.plotly_chart(parallel_figure(filtered, parameters, objectives, 'All Computed Solutions (Filtered)'),
                        use_container_width=True, key=key + '_plot')
        download_table(filtered, 'filtered_solutions.xlsx', key + '_export')


def render_prediction(bundle, values, reference, key):
    x = np.array([values])
    records = []
    for name, model in zip(bundle['objective_names'], bundle['models']):
        prediction = float(model.predict(x)[0])
        low, high = reference[name].min(), reference[name].max()
        normalized = 100 * (prediction - low) / (high - low) if high > low else 50.0
        records.append({'Objective': name, 'Estimated Value': prediction,
                        'Uncertainty estimate (±)': float(model.predict_uncertainty(x)[0]),
                        'Min Range': low, 'Max Range': high, 'Normalized (%)': normalized,
                        'Percentile': float((reference[name] <= prediction).mean() * 100)})
    table = pd.DataFrame(records)
    r = table['Normalized (%)'].tolist()
    theta = table['Objective'].tolist()
    hover = table[['Estimated Value', 'Min Range', 'Max Range', 'Percentile',
                   'Uncertainty estimate (±)']].to_numpy().tolist()
    fig = go.Figure(go.Scatterpolar(r=r + r[:1], theta=theta + theta[:1], fill='toself',
        line_color='#33966e', customdata=hover + hover[:1],
        hovertemplate='%{theta}<br>Prediction: %{customdata[0]:.3f}'
                      '<br>Reference range: %{customdata[1]:.3f} – %{customdata[2]:.3f}'
                      '<br>Normalized position: %{r:.1f}%<br>Percentile: %{customdata[3]:.1f}%'
                      '<br>Uncertainty estimate: ±%{customdata[4]:.3f}<extra></extra>'))
    fig.update_layout(polar=dict(radialaxis=dict(range=[min(0, min(r)), max(100, max(r))], ticksuffix='%')), height=450)
    st.plotly_chart(fig, use_container_width=True, key=key + '_radar')
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption('Radar values show position within the reference data range, not desirability. Uncertainty estimates are not calibrated confidence intervals.')
    training = bundle['training_data']
    outside = [p for p, v in zip(bundle['param_names'], values) if v < training[p].min() or v > training[p].max()]
    if outside:
        st.warning('Outside the training range: ' + ', '.join(outside))


def parameter_inputs(bundle, key):
    values = []
    columns = st.columns(3)
    for i, param in enumerate(bundle['param_names']):
        with columns[i % 3]:
            values.append(st.number_input(param, value=float(bundle['training_data'][param].median()), key=f'{key}_{i}'))
    return values


@st.fragment
def results_panel():
    run = st.session_state.get('optimization_run')
    if run is None:
        st.info('Run optimization in the Training tab to see results here.')
        return
    bundle, results = run['bundle'], run['results']
    params, objectives = bundle['param_names'], bundle['objective_names']
    front = results[results['Pareto']].sort_values('Weighted Sum', ascending=False)
    st.subheader('Optimization Results Explorer')
    st.write(f'{len(results):,} evaluated combinations · {len(front):,} Pareto solutions')
    st.caption('Pareto solutions are no worse in every active objective and better in at least one than any solution they dominate. Score ranks solutions separately; higher is better.')
    with st.expander('Optimization Result (Pareto Front)'):
        st.plotly_chart(parallel_figure(front, params, objectives, 'Solutions in Pareto Front'), use_container_width=True, key='pareto_plot')
        st.dataframe(front, use_container_width=True)
    with st.expander('Trend Analysis'):
        # Lazy rendering prevents closed trend diagrams from slowing every rerun.
        if st.checkbox('Show trend and sensitivity diagrams', key='show_trends'):
            render_trends(results, bundle, 'result_trends')
    with st.expander('Interactive Solution Explorer'):
        selected = st.selectbox('Select solution to explore:', list(front.index),
            format_func=lambda i: f'Solution {i + 1} (Score: {results.loc[i, "Weighted Sum"]:.3f})', key='selected_solution')
        row = results.loc[selected]
        st.dataframe(pd.DataFrame({'Parameter': params, 'Value': row[params].values}), hide_index=True)
        st.dataframe(pd.DataFrame({'Objective': objectives, 'Value': row[objectives].values,
            'Uncertainty (±)': [row['Uncertainty_' + o] for o in objectives], 'Weight': run['weights']}), hide_index=True)
        selected_fig = parallel_figure(front, params, objectives, 'Selected Solution')
        selected_fig.data[0].line.color = [1 if i == selected else 0 for i in front.index]
        selected_fig.data[0].line.colorscale = [[0, '#dddddd'], [1, '#16804a']]
        selected_fig.data[0].line.showscale = False
        st.plotly_chart(selected_fig, use_container_width=True, key='selected_solution_plot')
    with st.expander('Estimate Objectives Results for Custom Parameters Values'):
        with st.form('results_predict_form'):
            values = parameter_inputs(bundle, 'result_predict')
            submit = st.form_submit_button('Estimate Objectives Results')
        if submit:
            st.session_state['result_prediction'] = values
        if 'result_prediction' in st.session_state:
            render_prediction(bundle, st.session_state['result_prediction'], results, 'results_prediction')
    with st.expander('All Computed Solutions (Interactive)'):
        filter_results(results, params, objectives, 'results')


def model_overrides(objectives, mode, key):
    if mode.startswith('Manual'):
        return {o: st.selectbox(f'Model for {o}', ['Auto', 'Ridge', 'Random Forest', 'Hybrid'], key=f'{key}_{o}') for o in objectives}
    return {o: 'Auto' for o in objectives}


@st.fragment
def explorer_panel():
    st.subheader('Explorer')
    uploaded = st.file_uploader('Upload Optimization Results (.xlsx or .csv)', type=['xlsx', 'csv'], key='explorer_upload')
    if uploaded is None:
        st.info('Upload a results file to explore it.')
        return
    try:
        df = read_table(uploaded.getvalue(), uploaded.name)
        core = [c for c in df if not str(c).startswith('Uncertainty_') and c not in ('Weighted Sum', 'Pareto')]
        num = st.number_input('Enter Number of Objectives', 1, max(1, len(core) - 1), min(2, max(1, len(core) - 1)), key='explorer_nobj')
        train = df[core].copy()
        validate_training_data(train, num)
        params, objectives = core[:-num], core[-num:]
        data_key = hashlib.sha256(repr((compute_df_signature(train), num)).encode()).hexdigest()[:12]
        with st.expander('Setup Objectives Weights'):
            with st.form('explorer_weights_' + data_key):
                weights = [st.number_input(f'Weight for {o}', value=-1.0, key=f'exp_weight_{data_key}_{o}',
                    help='Negative = minimize, positive = maximize, zero = ignore.') for o in objectives]
                apply_weights = st.form_submit_button('Apply weights')
            slot = 'explorer_applied_weights_' + data_key
            if apply_weights or slot not in st.session_state:
                st.session_state[slot] = weights
            weights = st.session_state[slot]
        with st.expander('Dataset Overview'):
            st.write(f'{len(df):,} solutions; {len(params)} parameters; {len(objectives)} objectives.')
            st.dataframe(train.describe(), use_container_width=True)
        with st.expander('Estimate Objectives Results for Custom Parameters Values'):
            mode = st.radio('Model strategy for Explorer estimation', ['Auto (recommended)', 'Manual per objective'], key='explorer_mode')
            with st.form('explorer_predict_' + data_key):
                overrides = model_overrides(objectives, mode, 'explorer_override')
                values = parameter_inputs(dict(param_names=params, training_data=train), 'explorer_input_' + data_key)
                submitted = st.form_submit_button('Estimate Objectives Results')
            settings = dict(st.session_state['settings'], model_selection_mode='Auto' if mode.startswith('Auto') else 'Manual')
            signature = training_signature(train, num, settings, overrides)
            # All training output, including the progress message and smoke test, lives here.
            with st.expander('Training model verification', expanded=False):
                if submitted:
                    with st.spinner('Training surrogate models for Explorer estimation (per objective)...'):
                        bundle = get_training_bundle(train, num, settings, overrides, 'explorer_bundle')
                    st.session_state['explorer_prediction'] = (signature, values)
                bundle = st.session_state.get('explorer_bundle')
                if bundle and bundle['signature'] == signature:
                    render_verification(bundle, 'explorer_verify')
                else:
                    st.caption('Verification appears after estimating with the current dataset and settings.')
            saved = st.session_state.get('explorer_prediction')
            if saved and saved[0] == signature and bundle:
                render_prediction(bundle, saved[1], train, 'explorer_prediction')
            with st.expander('Trend Analysis', expanded=False):
                if st.checkbox('Show trend and sensitivity diagrams', key='explorer_show_trends_' + data_key):
                    current_bundle = bundle if bundle and bundle['signature'] == signature else dict(
                        param_names=params, objective_names=objectives)
                    render_trends(train, current_bundle, 'explorer_trends_' + data_key,
                                  source_label='uploaded solutions')
        with st.expander('All Computed Solutions (Interactive)'):
            display = df.copy()
            # Always recompute: uploaded scores may use old formulas or different weights.
            display['Weighted Sum'] = normalized_scores(display[objectives].to_numpy(), weights)
            weight_key = hashlib.sha256(repr(weights).encode()).hexdigest()[:8]
            filter_results(display, params, objectives, 'explorer_' + data_key + weight_key)
    except Exception as exc:
        st.error(str(exc))


def training_panel():
    st.subheader('Training and Optimization')
    num = st.number_input('Enter Number of Objectives', min_value=1, value=2, key='training_nobj')
    uploaded = st.file_uploader('Upload Training Data Excel file', type=['xlsx'], key='training_upload')
    if uploaded is None:
        st.info('Upload training data. Put parameter columns first and objective columns last.')
        return
    try:
        df = read_table(uploaded.getvalue(), uploaded.name)
        validate_training_data(df, num)
        params, objectives = list(df.columns[:-num]), list(df.columns[-num:])
        data_key = hashlib.sha256(repr((compute_df_signature(df), num)).encode()).hexdigest()[:12]
        settings = st.session_state['settings']
        with st.expander('Model Training and Validation', expanded=True):
            with st.form('model_form_' + data_key):
                overrides = model_overrides(objectives, settings['model_selection_mode'], 'training_override')
                train_clicked = st.form_submit_button('Train / update models')
            signature = training_signature(df, num, settings, overrides)
            if train_clicked:
                with st.spinner('Training and validating models...'):
                    get_training_bundle(df, num, settings, overrides, 'training_bundle')
            bundle = st.session_state.get('training_bundle')
            ready = bundle is not None and bundle['signature'] == signature
            if ready:
                render_verification(bundle, 'training_verify')
                with st.expander('Learning Curves'):
                    render_learning_curves(bundle, settings['cv_folds'], 'learning')
            else:
                st.info('Click Train / update models to use this dataset and the current model settings.')
        with st.expander('Verify prediction accuracy', expanded=False):
            render_accuracy_audit(df, num, settings, overrides, 'audit_' + data_key)
        with st.expander('Setup Training and run Optimization', expanded=False):
            st.caption('Edit the complete form, then submit once. Negative weights minimize, positive weights maximize; zero excludes an objective.')
            with st.form('optimization_form_' + data_key):
                ranges = {}
                columns = st.columns(3)
                for i, param in enumerate(params):
                    with columns[i % 3]:
                        st.markdown(f'**{param}**')
                        low = st.number_input('Min', value=float(df[param].min()), key=f'opt_min_{data_key}_{i}')
                        high = st.number_input('Max', value=float(df[param].max()), key=f'opt_max_{data_key}_{i}')
                        step = st.number_input('Step', value=max(0.1, (float(df[param].max()) - float(df[param].min())) / 10), key=f'opt_step_{data_key}_{i}')
                        ranges[param] = (low, high, step)
                st.subheader('Define Objective Weights')
                weights = [st.number_input(f'Weight — {obj}', value=-1.0, key=f'opt_weight_{data_key}_{i}') for i, obj in enumerate(objectives)]
                output_path = st.text_input('Output Excel or CSV file', value='evaluated_solutions.xlsx')
                submitted = st.form_submit_button('Optimize Parameters', disabled=not ready)
            if submitted:
                _, total = parameter_grid(ranges)
                st.write(f'Evaluating {total:,} combinations.')
                # A full results table is kept for the requested trends and filtering.
                if total > 1_000_000:
                    raise ValueError('This grid exceeds 1,000,000 combinations. Increase steps or narrow ranges before running.')
                start = time.perf_counter()
                progress = st.progress(0.0)
                results = evaluate_grid(bundle, ranges, weights, progress.progress)
                st.session_state['optimization_run'] = dict(bundle=bundle, results=results, weights=weights, ranges=ranges)
                for key in ('result_prediction', 'results_filters', 'selected_solution', 'results_export_prepared'):
                    st.session_state.pop(key, None)
                try:
                    target, fallback_reason = save_table_export(results, output_path)
                    st.session_state['optimization_notice'] = f'Saved {len(results):,} solutions to {target}. {int(results.Pareto.sum()):,} Pareto solutions. Completed in {time.perf_counter() - start:.1f}s. Open Results to explore.'
                    if fallback_reason:
                        st.session_state['optimization_notice'] += f' {fallback_reason} Saved as CSV instead.'
                except Exception as exc:
                    st.session_state['optimization_notice'] = f'Optimization completed, but export failed: {exc}. Results remain available in this session.'
                st.rerun()
            if 'optimization_notice' in st.session_state:
                st.info(st.session_state['optimization_notice'])
    except Exception as exc:
        st.error(str(exc))


def main():
    st.set_page_config(page_title='WSGT_PAMO Machine Learning', layout='wide', initial_sidebar_state='collapsed')
    from packaging.version import Version
    if Version(st.__version__) < Version("1.49.0"):
        import sys
        st.error(f"PAMO requires Streamlit 1.49 or newer. This Python environment uses {st.__version__}.")
        st.write("Stop PAMO, run this command in a terminal, then start PAMO again using the same Python:")
        st.code(f'"{sys.executable}" -m pip install --upgrade "streamlit>=1.49,<2"', language="text")
        st.caption("The newer version is needed for Explorer's nested verification panel. Your training data is unchanged.")
        st.stop()
    assets = Path(__file__).resolve().parent
    with st.sidebar:
        if (assets / 'Pamo_Icon_Black.png').exists():
            st.image(str(assets / 'Pamo_Icon_Black.png'), width=80)
        st.markdown('## WSGT_PAMO')
        st.caption(f'Version {VERSION}')
        with st.expander('Setup'):
            with st.form('settings_form'):
                alpha = st.slider('Ridge alpha (regularization)', 0.001, 10.0, 0.1, 0.001)
                trees = st.slider('Number of Trees in Random Forest', 1, 200, 50)
                mode = st.radio('Model strategy', ['Auto (recommended)', 'Manual per objective'])
                folds = st.slider('Cross-Validation Folds', 3, 20, 5)
                cores = st.slider('Number of CPU Cores', 1, max(2, os.cpu_count() or 2), min(8, os.cpu_count() or 1))
                apply = st.form_submit_button('Apply model settings')
            if apply or 'settings' not in st.session_state:
                st.session_state['settings'] = dict(ridge_alpha=alpha, rf_n_estimators=trees, cv_folds=folds,
                    num_cores=cores, model_selection_mode='Auto' if mode.startswith('Auto') else 'Manual')
        st.markdown('---')
        st.caption('A product of Werner Sobek Green Technologies GmbH · Fachgruppe Simulation')
        st.caption('Coded by Rodrigo Carvalho')
    if (assets / 'WS_Logo.png').exists():
        st.image(str(assets / 'WS_Logo.png'), width=400)
    st.title('Machine Learning Assisted Optimization')
    training, results, explorer = st.tabs(['Training', 'Results', 'Explorer'])
    with training:
        training_panel()
    with results:
        results_panel()
    with explorer:
        explorer_panel()


if __name__ == '__main__':
    main()
