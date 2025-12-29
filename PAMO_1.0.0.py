import streamlit as st
import pandas as pd
import numpy as np
from PIL.ImageChops import lighter
from deap import base, creator, tools
import itertools
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import pickle
import tempfile
from typing import Optional
from functools import partial
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.cm as cm


# --- Data signature helper (for reliable caching) ---
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


# --- Poly(deg=2)+Ridge helpers (PAMO 1.1.14) ---

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



class HybridEstimator(BaseEstimator, RegressorMixin):
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

st.set_page_config(
    initial_sidebar_state="collapsed",  # Optional
    page_title="WSGT_PAMO Machine Learning",
    page_icon="Pamo_Icon_White.png",
    layout="wide"
)

st.sidebar.image("Pamo_Icon_Black.png", width=80)
st.sidebar.write("## WSGT_PAMO")
st.sidebar.write("Version 1.1.10")
st.sidebar.write("Dynamic Solver + Cache")
st.sidebar.markdown("---")

col1, col2 = st.columns(2)

with col2:
    st.image("WS_Logo.png", width=900)

st.title("Machine Learning Assisted Optimization")


# Create a multi-objective fitness class with variable number of objectives
def create_fitness_class(num_objectives, weights):
    if 'FitnessMulti' in creator.__dict__:
        del creator.FitnessMulti
    creator.create("FitnessMulti", base.Fitness, weights=tuple(weights))
    return creator.FitnessMulti


# Create an individual class
def create_individual_class(num_objectives, weights):
    if 'Individual' in creator.__dict__:
        del creator.Individual
    creator.create("Individual", list, fitness=create_fitness_class(num_objectives, weights), uncertainty=list)


# Generate parameter combinations - lazy evaluation
def parameter_combinations_generator(param_ranges):
    """Generate parameter combinations on-demand without storing all in memory.

    param_ranges: dict {param_name: (min_val, max_val, step)}
    Returns: (generator, total_combinations)
    """
    param_values = []
    for min_val, max_val, step in param_ranges.values():
        if float(min_val) == float(max_val):
            param_values.append([float(min_val)])
        else:
            min_val = float(min_val)
            max_val = float(max_val)
            step = float(step)
            if step <= 0:
                param_values.append([min_val])
            else:
                param_values.append(np.arange(min_val, max_val + step / 2.0, step))

    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    return itertools.product(*param_values), total_combinations

# Function to load data from Excel file
def load_data(num_objectives):
    uploaded_file = st.file_uploader("Upload Training Data Excel file", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            if len(df.columns) < num_objectives:
                st.error(
                    f"Number of objectives ({num_objectives}) exceeds number of columns in the file ({len(df.columns)}).")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None


# (preprocess_and_train_with_cv is defined later with per-objective auto model selection)


def visualize_model_performance(X, y, estimators, objective_names, cv_folds=5):
    """
    Plot learning curves (R²) for each objective.

    Why this sometimes failed on Windows/Streamlit:
      - sklearn.learning_curve uses joblib for parallel CV when n_jobs > 1.
      - In some Windows/Streamlit setups, process-based parallelism can raise OSError: [Errno 22] Invalid argument.
    Mitigation:
      - Prefer a thread-based backend for learning curves.
      - If it still fails, fall back to single-thread execution.
    """
    try:
        if X is None or y is None or not estimators:
            st.warning("No models or data available for learning curves.")
            return

        # y is expected as list/array of per-objective vectors
        n_obj = len(y)
        if n_obj == 0:
            st.warning("No objectives available for learning curves.")
            return

        fig, axes = plt.subplots(1, n_obj, figsize=(4.2 * n_obj, 4.2), frameon=False)
        if n_obj == 1:
            axes = [axes]

        # CV folds cannot exceed the sample count
        cv_folds_int = int(cv_folds) if cv_folds is not None else 5
        cv_folds_int = min(max(2, cv_folds_int), len(X))

        # Training sizes: fractions are OK; keep count small for speed
        train_sizes = np.linspace(0.2, 1.0, min(8, max(2, len(X))))

        # Use thread backend to avoid Windows process-spawn issues inside Streamlit
        n_jobs_req = max(1, int(st.session_state.get('num_cores', 1)))

        from joblib import parallel_backend

        for i, (y_i, est, ax, obj_name) in enumerate(zip(y, estimators, axes, objective_names)):
            try:
                with parallel_backend("threading", n_jobs=n_jobs_req):
                    ts, train_scores, test_scores = learning_curve(
                        est,
                        X,
                        y_i,
                        cv=cv_folds_int,
                        n_jobs=n_jobs_req,
                        train_sizes=train_sizes,
                        scoring="r2"
                    )
            except Exception:
                # Fall back to single-threaded execution as a last resort
                ts, train_scores, test_scores = learning_curve(
                    est,
                    X,
                    y_i,
                    cv=cv_folds_int,
                    n_jobs=1,
                    train_sizes=train_sizes,
                    scoring="r2"
                )

            try:
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)

                ax.plot(ts, train_mean, marker="o", label="Training R²")
                ax.fill_between(ts, train_mean - train_std, train_mean + train_std, alpha=0.15)

                ax.plot(ts, test_mean, marker="o", label="CV R²")
                ax.fill_between(ts, test_mean - test_std, test_mean + test_std, alpha=0.15)

                ax.set_title(f"Learning Curve - {obj_name}")
                ax.set_xlabel("Training samples")
                ax.set_ylabel("R²")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(loc="best", fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, f"Plot error: {str(e)}", ha="center", va="center")
                ax.set_title(f"Learning Curve - {obj_name} (Plot error)")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error visualizing learning curves: {e}")


def evaluate_batch(batch_params, models, num_objectives, objective_uncertainties=None):
    results = []

    for params in batch_params:
        params_array = np.array([params])

        predictions = []
        uncertainties = []

        for obj_idx, model in enumerate(models):
            # Prediction
            try:
                pred_val = float(model.predict(params_array)[0])
            except Exception:
                pred_val = 0.0

            # Uncertainty: hybrid if available; fallback to constant list
            if hasattr(model, "predict_uncertainty"):
                try:
                    unc_val = float(np.asarray(model.predict_uncertainty(params_array))[0])
                except Exception:
                    unc_val = 0.0
            elif objective_uncertainties is not None and obj_idx < len(objective_uncertainties):
                unc_val = float(objective_uncertainties[obj_idx])
            else:
                unc_val = 0.0

            predictions.append(pred_val)
            uncertainties.append(unc_val)

        # Ensure correct length
        if len(predictions) != num_objectives:
            if len(predictions) < num_objectives:
                predictions = predictions + [0.0] * (num_objectives - len(predictions))
            else:
                predictions = predictions[:num_objectives]

        if len(uncertainties) != num_objectives:
            if len(uncertainties) < num_objectives:
                uncertainties = uncertainties + [0.0] * (num_objectives - len(uncertainties))
            else:
                uncertainties = uncertainties[:num_objectives]

        rounded_fitness = [f"{float(fit_val):.2f}" for fit_val in predictions]
        rounded_uncertainty = [f"{float(unc_val):.2f}" for unc_val in uncertainties]

        results.append({
            'params': params,
            'predictions': predictions,
            'uncertainties': uncertainties,
            'rounded_fitness': rounded_fitness,
            'rounded_uncertainty': rounded_uncertainty
        })

    return results


def process_completed_results(async_results, hof, temp_results_file,
                              processed_count, total_combinations,
                              progress_bar, status_text, start_time):
    """Process any completed async results"""
    # Check which results are ready
    completed_indices = []
    for i, (future, batch_size) in enumerate(async_results):
        if future.done():
            completed_indices.append(i)

            try:
                # Get the result
                result_batch = future.result()
            except Exception as e:
                st.error(f"Error in batch processing: {e}")
                continue

            # Process results and update Pareto front
            with open(temp_results_file, 'a') as f:
                for result in result_batch:
                    # Create individual
                    individual = creator.Individual(result['params'])
                    individual.fitness.values = tuple(result['predictions'])
                    individual.weighted_sum = 0.0  # to be calculated after normalization
                    individual.uncertainty = tuple(result['uncertainties'])

                    # Update Pareto front
                    hof.update([individual])

                    # Write to CSV directly to save memory
                    row_data = [str(float(p)) for p in result['params']] + result['rounded_fitness'] + result[
                        'rounded_uncertainty']
                    f.write(','.join(row_data) + '\n')

            # Update progress
            processed_count += batch_size
            progress = min(1.0, processed_count / total_combinations)
            progress_bar.progress(progress)

            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time

            status_text.text(f"Processed {processed_count}/{total_combinations} combinations. " +
                             f"Elapsed: {elapsed_time:.2f}s. Estimated remaining: {remaining_time:.2f}s.")

    # Remove completed results
    for i in sorted(completed_indices, reverse=True):
        async_results.pop(i)

    return processed_count


# Function to perform streaming optimization with optimized parallel processing
@st.cache_resource
def optimize_parameters_parallel(param_ranges, num_objectives, _models, weights, output_path, objective_names, objective_uncertainties=None, models_signature: str = "", num_cores: int = 1):
    try:
        st.write("Starting optimization process with streaming parallel processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        hof = tools.ParetoFront()

        # Get parameter combinations generator and total count
        param_generator, total_combinations = parameter_combinations_generator(param_ranges)

        st.write(f"Total parameter combinations to evaluate: {total_combinations}")

        # Threads for parallel evaluation (caps to available CPU cores)
        requested_cores = int(num_cores) if num_cores is not None else int(st.session_state.get('num_cores', 8))
        available_cores = os.cpu_count() or requested_cores
        num_cores = max(1, min(requested_cores, available_cores))
        st.write(f"Using {num_cores} threads for processing (requested: {requested_cores}, available: {available_cores})")

        # Calculate optimal batch size for Streamlit Cloud (smaller batches for stability)
        if total_combinations < 1000:
            batch_size = max(1, min(250, total_combinations // 10))
        elif total_combinations < 10000:
            batch_size = max(5, min(500, total_combinations // 20))
        else:
            batch_size = max(10, min(1000, total_combinations // 50))  # Much smaller batches for large datasets

        st.write(f"Using adaptive batch size: {batch_size}")

        # Prepare for streaming processing
        processed_count = 0
        batch_count = 0

        # Prepare column labels for the DataFrame
        param_labels = list(param_ranges.keys())
        fitness_labels = list(objective_names)
        uncertainty_labels = [f'Uncertainty_{name}' for name in objective_names]

        # Create DataFrame for results with chunked writing
        df_columns = param_labels + fitness_labels + uncertainty_labels

        # Create a temporary file for results
        temp_results_file = "temp_results.csv"
        with open(temp_results_file, 'w') as f:
            # Write header
            f.write(','.join(df_columns) + '\n')

        # Create a pool of worker threads (more Streamlit-friendly than processes)
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Use a queue of async results to maximize CPU utilization
            async_results = []
            batches = []
            current_batch = []

            # Process parameter combinations in streaming batches
            for i, params in enumerate(param_generator):
                current_batch.append(params)

                # When batch is full, submit it asynchronously
                if len(current_batch) >= batch_size:
                    batch_count += 1
                    batches.append(current_batch)

                    # Submit batch for async processing
                    future = executor.submit(evaluate_batch, current_batch, _models, num_objectives, objective_uncertainties)
                    async_results.append((future, len(current_batch)))

                    # Reset current batch
                    current_batch = []

                # Process completed results to free up memory
                processed_count = process_completed_results(
                    async_results, hof, temp_results_file,
                    processed_count, total_combinations,
                    progress_bar, status_text, start_time
                )

                # Periodic memory cleanup and progress check
                if i % 100 == 0:
                    import gc
                    gc.collect()
            # Submit any remaining combinations
            if current_batch:
                batch_count += 1
                batches.append(current_batch)

                # Submit batch for async processing
                future = executor.submit(evaluate_batch, current_batch, _models, num_objectives, objective_uncertainties)
                async_results.append((future, len(current_batch)))

            # Wait for all remaining results to complete
            while async_results:
                processed_count = process_completed_results(
                    async_results, hof, temp_results_file,
                    processed_count, total_combinations,
                    progress_bar, status_text, start_time
                )
                time.sleep(0.1)  # Short sleep to prevent CPU spinning

        # Complete progress bar
        progress_bar.progress(1.0)

        # Read results from temporary file
        results_df = pd.read_csv(temp_results_file, encoding='latin-1')

        # Calculate weighted sum for each solution
        if 'Weighted Sum' not in results_df.columns:
            # Normalize and calculate weighted sum
            weighted_sum = np.zeros(len(results_df))
            for i, (obj, w) in enumerate(zip(fitness_labels, weights)):
                col_values = results_df[obj].astype(float)
                v_min = col_values.min()
                v_max = col_values.max()

                if v_max == v_min:
                    normalized = np.zeros_like(col_values)
                elif w < 0:
                    normalized = w * (v_max - col_values) / (v_max - v_min)
                else:
                    normalized = w * (col_values - v_min) / (v_max - v_min)

                weighted_sum += normalized

            results_df['Weighted Sum'] = weighted_sum
            for ind, wsum in zip(hof, weighted_sum):
                ind.weighted_sum = wsum

        # Save results to Excel file if not too large
        try:
            if len(results_df) <= 1048576:  # Excel row limit
                results_df.to_excel(output_path, index=False)
                st.success(f"Results saved to {output_path}")
                # Store the output path in session state for later use
                st.session_state['output_path'] = output_path
            else:
                csv_path = output_path.replace('.xlsx', '.csv')
                results_df.to_csv(csv_path, index=False)
                st.warning(f"Results too large for Excel. Saved to {csv_path} instead.")
                # Store the CSV path in session state for later use
                st.session_state['output_path'] = csv_path
        except Exception as e:
            st.error(f"Error saving results: {e}")
            # Fallback to CSV
            try:
                csv_path = output_path.replace('.xlsx', '.csv')
                results_df.to_csv(csv_path, index=False)
                st.success(f"Results saved to {csv_path}")
                # Store the CSV path in session state for later use
                st.session_state['output_path'] = csv_path
            except:
                st.error("Failed to save results to file.")

        # Clean up temporary file
        try:
            os.remove(temp_results_file)
        except:
            pass

        # Display optimization summary
        st.write(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        st.write(f"Evaluated {processed_count} parameter combinations")
        st.write(f"Found {len(hof)} solutions on the Pareto front")

        return hof

    except Exception as e:
        st.error(f"Error in optimization: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# Helper function to normalize weighted sum consistently
def normalize_weighted_sum(objective_values, weights):
    """Calculate normalized weighted sum for given objective values and weights"""
    weighted_sum = 0.0

    for i, (obj_vals, w) in enumerate(zip(objective_values.T, weights)):
        v_min = obj_vals.min()
        v_max = obj_vals.max()

        if v_max == v_min:
            normalized = np.zeros_like(obj_vals)
        elif w < 0:
            normalized = w * (v_max - obj_vals) / (v_max - v_min)
        else:
            normalized = w * (obj_vals - v_min) / (v_max - v_min)

        weighted_sum += normalized

    return weighted_sum


# Helper function to create dimension for parallel coordinates plot with non-overlapping tick labels
def create_dimension(label, values, min_val, max_val):
    """Create a dimension for parallel coordinates with clean tick labels"""
    # Let Plotly handle tick positioning automatically for better readability
    # Only specify range and let Plotly determine optimal tick positions
    if min_val == max_val:
        # Handle edge case where all values are the same
        return dict(
            range=[min_val - 0.1, max_val + 0.1],
            label=label,
            values=values
        )
    else:
        return dict(
            range=[min_val, max_val],
            label=label,
            values=values
        )


# Visualize Pareto front using Plotly for interactivity
def visualize_pareto_front(hof, objective_names_list, param_names_list, weights):
    try:
        # Ensure objective_names_list and param_names_list are lists
        objective_names_list = list(objective_names_list)
        param_names_list = list(param_names_list)

        # Extract objective values for all solutions
        objective_values = np.array([list(ind.fitness.values) for ind in hof])

        # Calculate weighted sum for each solution
        weighted_sums = np.array([ind.weighted_sum for ind in hof])

        # Create a DataFrame for the parallel coordinates plot
        data = []
        for i, ind in enumerate(hof):
            solution_data = {}
            # Add parameter values
            for j, param_name in enumerate(param_names_list):
                solution_data[param_name] = float(ind[j])

            # Add objective values
            for j, obj_name in enumerate(objective_names_list):
                solution_data[obj_name] = float(ind.fitness.values[j])

            # Add weighted sum and solution ID
            solution_data['Weighted Sum'] = float(weighted_sums[i])
            solution_data['Solution ID'] = i

            data.append(solution_data)

        df = pd.DataFrame(data)

        # Create parallel coordinates plot with Plotly
        dimensions = []

        # Add parameter dimensions with non-overlapping tick labels
        for param in param_names_list:
            param_min = df[param].min()
            param_max = df[param].max()
            dimensions.append(create_dimension(param, df[param], param_min, param_max))

        # Add objective dimensions with non-overlapping tick labels
        for obj in objective_names_list:
            obj_min = df[obj].min()
            obj_max = df[obj].max()
            dimensions.append(create_dimension(obj, df[obj], obj_min, obj_max))

        # Add weighted sum dimension at the far right
        ws_min = df['Weighted Sum'].min()
        ws_max = df['Weighted Sum'].max()
        dimensions.append(create_dimension('Weighted Sum', df['Weighted Sum'], ws_min, ws_max))

        # Create the parallel coordinates plot
        with st.expander("Optimization Result (Pareto Front)", expanded=False):
            st.write("## Solutions in Pareto Front")
            st.write("Here you find the non-dominated solutions in Pareto Front")
            fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df['Weighted Sum'],
                    colorscale='Tealrose',
                    showscale=True,
                    colorbar=dict(title='Weighted Sum', nticks=11)
                ),
                dimensions=dimensions,
                unselected=dict(line=dict(opacity=0)),
                rangefont=dict(family="Arial", size=14),
                tickfont=dict(family="Arial", size=11, weight="bold", color="black"),

            )
            )

            fig.update_layout(
                title="Solutions in Pareto Front",
                height=600,
                margin=dict(l=50, r=50, t=100, b=50),  # Increased left and right margins for axis labels
                font=dict(family='Arial', size=14)  # Reduced font size to prevent overlapping
            )

            # Display the interactive plot
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error visualizing Pareto front: {e}")
        import traceback
        st.code(traceback.format_exc())


# Visualize a single selected solution
def visualize_selected_solution(solution, param_names, objective_names, weights, hof):
    try:
        st.write("## Interactive Solution Explorer")
        st.write("Here you can isolate and investigate a specific non-dominated solutions from Pareto Front")
        st.subheader("Selected Solution Details")

        # Ensure param_names and objective_names are lists
        param_names = list(param_names)
        objective_names = list(objective_names)

        # Create tables for parameter and objective values (formatted to exactly 2 decimal places)
        col1, col2 = st.columns(2)

        with col1:
            st.write("Parameter Values")
            param_df = pd.DataFrame({
                'Parameter': param_names,
                'Value': [f"{float(p):.2f}" for p in solution]  # Format to exactly 2 decimal places
            })
            st.table(param_df)

        with col2:
            st.write("Objective Values")
            obj_df = pd.DataFrame({
                'Objective': objective_names,
                'Value': [f"{float(f):.2f}" for f in solution.fitness.values],  # Format to exactly 2 decimal places
                '(±)': [f"{float(u):.2f}" for u in solution.uncertainty],
                # Format to exactly 2 decimal places
                'Weight': [f"{float(w):.2f}" for w in weights]  # Format to exactly 2 decimal places
            })
            st.table(obj_df)

        # Create the parallel coordinates plot for the selected solution using Plotly for consistency
        # Combine parameter values and objective values for visualization
        all_names = param_names + objective_names

        # Extract all data for all solutions (parameters + objectives)
        all_data = []
        for ind in hof:
            solution_data = {}
            # Add parameter values
            for j, param_name in enumerate(param_names):
                solution_data[param_name] = float(ind[j])

            # Add objective values
            for j, obj_name in enumerate(objective_names):
                solution_data[obj_name] = float(ind.fitness.values[j])

            # Add solution ID
            solution_data['is_selected'] = 0  # Not selected

            all_data.append(solution_data)

        # Add selected solution data
        selected_data = {}
        # Add parameter values
        for j, param_name in enumerate(param_names):
            selected_data[param_name] = float(solution[j])

        # Add objective values
        for j, obj_name in enumerate(objective_names):
            selected_data[obj_name] = float(solution.fitness.values[j])

        # Mark as selected
        selected_data['is_selected'] = 1  # Selected

        # Create DataFrame with all solutions
        df_all = pd.DataFrame(all_data)
        df_selected = pd.DataFrame([selected_data])

        # Combine all data
        df_combined = pd.concat([df_all, df_selected])

        # Create dimensions for parallel coordinates with non-overlapping tick labels
        dimensions = []
        for name in all_names:
            name_min = df_combined[name].min()
            name_max = df_combined[name].max()
            dimensions.append(create_dimension(name, df_combined[name], name_min, name_max))

        # Create figure
        fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_combined['is_selected'],
                colorscale=[[0, 'rgba(250,250,250,0.5)'], [1, 'rgba(255,0,0,1)']],
                showscale=False
            ),
            dimensions=dimensions,
            unselected=dict(line=dict(opacity=0)),
            rangefont=dict(family="Arial", size=14),
            tickfont=dict(family="Arial", size=12, weight="bold", color="black")
        )
        )

        fig.update_layout(
            title="Selected Solution",
            height=600,
            margin=dict(l=50, r=50, t=100, b=50),  # Increased margins for labels
            font=dict(family="Arial", size=14)  # Smaller font to prevent overlap
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error visualizing selected solution: {e}")
        import traceback
        st.code(traceback.format_exc())


# Visualize parameter trends
def visualize_parameter_trends(hof, param_names, objective_names):
    with st.expander("Trend Analysis", expanded=False):
        try:
            st.write("## Trend Analysis")
            st.write("Here you can investigate how each parameter affects objectives")
            # Ensure param_names and objective_names are lists
            param_names = list(param_names)
            objective_names = list(objective_names)

            # Extract data for all solutions
            param_data = np.zeros((len(hof), len(param_names)))
            objective_data = np.zeros((len(hof), len(objective_names)))

            for i, ind in enumerate(hof):
                # Add parameter values
                for j, param in enumerate(ind):
                    param_data[i, j] = float(param)

                # Add objective values
                for j, obj in enumerate(ind.fitness.values):
                    objective_data[i, j] = float(obj)

            # Create a trend diagram for each parameter
            for p_idx, param_name in enumerate(param_names):
                st.subheader(f"Parameter: {param_name}")

                # Create a figure with subplots for each objective (without frame)
                fig, axes = plt.subplots(1, len(objective_names), figsize=(15, 4), frameon=False)
                if len(objective_names) == 1:
                    axes = [axes]

                for o_idx, (obj_name, ax) in enumerate(zip(objective_names, axes)):
                    # Extract parameter and objective values
                    x = param_data[:, p_idx]
                    y = objective_data[:, o_idx]

                    # Plot scatter points
                    ax.scatter(x, y, alpha=0.5, color='green')

                    # Add trendline if there are enough points
                    if len(x) > 1:
                        try:
                            # Fit a polynomial of degree 1 (linear) or 2 (quadratic) based on data size
                            degree = 2 if len(x) > 5 else 1
                            z = np.polyfit(x, y, degree)
                            p = np.poly1d(z)

                            # Generate x values for the trendline
                            x_trend = np.linspace(min(x), max(x), 100)

                            # Plot the trendline
                            ax.plot(x_trend, p(x_trend), "r--")
                        except:
                            # If fitting fails, skip trendline
                            pass

                    ax.set_xlabel(param_name)
                    ax.set_ylabel(obj_name)
                    ax.set_title(f"{param_name} vs {obj_name}")
                    ax.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error visualizing parameter trends: {e}")
            import traceback
            st.code(traceback.format_exc())


# Initialize session state variables
def initialize_session_state():
    if 'models' not in st.session_state:
        st.session_state['models'] = []

    if 'optimization_complete' not in st.session_state:
        st.session_state['optimization_complete'] = False

    if 'hof' not in st.session_state:
        st.session_state['hof'] = None

    if 'param_names' not in st.session_state:
        st.session_state['param_names'] = []

    if 'objective_names' not in st.session_state:
        st.session_state['objective_names'] = []

    if 'weights' not in st.session_state:
        st.session_state['weights'] = []

    if 'selected_solution_index' not in st.session_state:
        st.session_state['selected_solution_index'] = 0

    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = "Training"

    if 'num_cores' not in st.session_state:
        st.session_state['num_cores'] = 8

    if 'output_path' not in st.session_state:
        st.session_state['output_path'] = "evaluated_solutions.xlsx"


# Callback for solution selection
def on_solution_select():
    # This function will be called when a solution is selected
    # Get the key that triggered the callback
    for key in st.session_state:
        if key.startswith('solution_selector_') and st.session_state[key] != st.session_state.get(
                'selected_solution_index', 0):
            st.session_state['selected_solution_index'] = st.session_state[key]
            break


# Function to display optimization results in a separate mode
def display_optimization_results():
    st.title("Optimization Results Explorer")
    st.write("Explore optimization results")

    # Get data from session state
    hof = st.session_state['hof']
    param_names_list = st.session_state['param_names']
    objective_names_list = st.session_state['objective_names']
    weights = st.session_state['weights']
    if not hof:
        st.error("No optimization results found.")
        return

    # Display Pareto front visualization
    visualize_pareto_front(hof, objective_names_list, param_names_list, weights)

    # Display Parameter Trend Analysis (only once)
    visualize_parameter_trends(hof, param_names_list, objective_names_list)

    # Recalculate normalized weighted sum for consistent display
    objective_values = np.array([list(ind.fitness.values) for ind in hof])
    normalized_weighted_sums = normalize_weighted_sum(objective_values, weights)

    # Update the weighted sum in each individual
    for ind, norm_wsum in zip(hof, normalized_weighted_sums):
        ind.weighted_sum = norm_wsum

    # Sort solutions by normalized weighted sum (lowest to highest)
    sorted_solutions = sorted(range(len(hof)), key=lambda i: hof[i].weighted_sum, reverse=False)

    # Create a mappin
# --- Model selection (Auto/Manual) helpers (PAMO 1.1.14) ---

DEFAULT_RIDGE_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_RF_TREES_FINAL = 150
DEFAULT_RF_TREES_CV = 120
DEFAULT_RF_MIN_SAMPLES_LEAF = 2

SIMPLE_MODEL_TIE_TOL = 0.02      # 2%: if Ridge is within 2% of best NMAE, prefer Ridge
HYBRID_MIN_GAIN_TOL = 0.01       # 1%: require Hybrid to beat best single model by >1% to justify complexity


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


# Light-weight RF hyperparameter tuning (kept small for interactive use)
DEFAULT_RF_GRID = [
    {"min_samples_leaf": 1, "max_depth": None, "max_features": "sqrt"},
    {"min_samples_leaf": 2, "max_depth": None, "max_features": "sqrt"},
    {"min_samples_leaf": 4, "max_depth": None, "max_features": "sqrt"},
    {"min_samples_leaf": 1, "max_depth": 12, "max_features": "sqrt"},
    {"min_samples_leaf": 2, "max_depth": 12, "max_features": "sqrt"},
    {"min_samples_leaf": 4, "max_depth": 12, "max_features": "sqrt"},
]

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

# Function to preprocess data and train model with cross-validation
@st.cache_resource(show_spinner=False, hash_funcs={pd.DataFrame: compute_df_signature})
def preprocess_and_train_with_cv(df, num_objectives, ridge_alpha=0.1, cv_folds=5,
                                 model_selection_mode="Auto", objective_model_overrides=None, store_in_session: bool = True, rf_n_estimators: int = 150, num_cores: int = 8, data_signature: str = ""):
    """
    Train one surrogate per objective, with per-objective model selection.

    Candidate models (per objective):
      - Ridge: Poly(deg=2)+Scaler+RidgeCV (auto) or Ridge(alpha) (manual)
      - Random Forest
      - Hybrid: Ridge baseline + RF residual corrector

    Selection criterion:
      - Primary: CV NMAE (MAE normalized by target range)
      - Reported: CV R² + MAE + NMAE
      - Bias: prefer Ridge if within SIMPLE_MODEL_TIE_TOL of best;
              choose Hybrid only if it beats best single model by > HYBRID_MIN_GAIN_TOL

    Manual override:
      objective_model_overrides can specify "Ridge", "Random Forest", "Hybrid", or "Auto" per objective name.
    """
    try:
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

            # Print short summary (keeps existing behavior of showing training diagnostics)
            st.write(f"Objective {i + 1} ({obj_name}) -> Selected: **{chosen}**")
            st.write(f"CV NMAE: Ridge={ridge_info['nmae']:.3f} | RF={rf_info['nmae']:.3f} | Hybrid={hybrid_info['nmae']:.3f}")
            if np.isfinite(chosen_info['r2']):
                st.write(f"Selected CV R²: {chosen_info['r2']:.2f} (folds: {[f'{s:.2f}' if np.isfinite(s) else 'nan' for s in chosen_info['fold_scores']]})")
            st.write(f"Estimated uncertainty floor (CV residual std): {sigma_floor:.2f}")
            st.markdown("---")

        # CV boxplot (selected model per objective)
        if cv_scores:
            fig, ax = plt.subplots(figsize=(10, 6), frameon=False)
            ax.boxplot(cv_scores)
            ax.set_xticklabels(objective_names, rotation=45, ha='right')
            ax.set_ylabel("R² Score")
            ax.set_title("Cross-Validation Performance by Objective (Selected Model)")
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

        # Feature sensitivity plot (parameter-level)
        if feature_importances:
            fig, axes = plt.subplots(1, len(y_list), figsize=(15, 5), sharey=False, frameon=False)
            if len(y_list) == 1:
                axes = [axes]

            for i, (ax, importances) in enumerate(zip(axes, feature_importances)):
                importances = np.asarray(importances, dtype=float).reshape(-1)
                if len(importances) != len(X.columns):
                    importances = np.resize(importances, len(X.columns))
                indices = np.argsort(importances)
                sorted_importances = importances[indices]
                sorted_names = [X.columns[j] for j in indices]

                max_imp = float(np.max(sorted_importances)) if len(sorted_importances) else 0.0
                colors = cm.RdYlGn_r(sorted_importances / max_imp if max_imp > 0 else 0)

                y_pos = np.arange(len(indices))
                ax.barh(y_pos, sorted_importances, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(sorted_names)
                ax.set_title(f"Sensitivity - {objective_names[i]} ({chosen_model_kinds[i]})")
                ax.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            st.pyplot(fig)

        # Store outputs
        if store_in_session:
            st.session_state['models'] = models
            st.session_state['objective_uncertainties'] = objective_uncertainties
            st.session_state['model_kinds'] = chosen_model_kinds
            st.session_state['objective_model_kinds'] = chosen_model_kinds
            st.session_state['objective_model_reports'] = per_objective_reports
            st.session_state['plot_estimators'] = plot_estimators
            st.session_state['models_signature'] = f"{data_signature}|alpha={ridge_alpha}|folds={cv_folds}|trees={rf_n_estimators}|cores={num_cores}|kinds={tuple(chosen_model_kinds)}"

        # Quick prediction smoke test
        if len(X) > 0:
            sample = X.iloc[0].values
            st.write("### Testing Model Predictions")
            st.write(f"Sample input: {sample}")
            for i, model in enumerate(models):
                pred = float(np.asarray(model.predict([sample]))[0])
                st.write(f"Objective {i + 1} ({objective_names[i]}) prediction: {pred:.2f}")

        return models, X.columns, objective_names

    except Exception as e:
        st.error(f"Error in model training: {e}")
        import traceback
        st.code(traceback.format_exc())


def evaluate_batch(batch_params, models, num_objectives, objective_uncertainties=None):
    results = []

    for params in batch_params:
        params_array = np.array([params])

        predictions = []
        uncertainties = []

        for obj_idx, model in enumerate(models):
            # Prediction
            try:
                pred_val = float(model.predict(params_array)[0])
            except Exception:
                pred_val = 0.0

            # Uncertainty: hybrid if available; fallback to constant list
            if hasattr(model, "predict_uncertainty"):
                try:
                    unc_val = float(np.asarray(model.predict_uncertainty(params_array))[0])
                except Exception:
                    unc_val = 0.0
            elif objective_uncertainties is not None and obj_idx < len(objective_uncertainties):
                unc_val = float(objective_uncertainties[obj_idx])
            else:
                unc_val = 0.0

            predictions.append(pred_val)
            uncertainties.append(unc_val)

        # Ensure correct length
        if len(predictions) != num_objectives:
            if len(predictions) < num_objectives:
                predictions = predictions + [0.0] * (num_objectives - len(predictions))
            else:
                predictions = predictions[:num_objectives]

        if len(uncertainties) != num_objectives:
            if len(uncertainties) < num_objectives:
                uncertainties = uncertainties + [0.0] * (num_objectives - len(uncertainties))
            else:
                uncertainties = uncertainties[:num_objectives]

        rounded_fitness = [f"{float(fit_val):.2f}" for fit_val in predictions]
        rounded_uncertainty = [f"{float(unc_val):.2f}" for unc_val in uncertainties]

        results.append({
            'params': params,
            'predictions': predictions,
            'uncertainties': uncertainties,
            'rounded_fitness': rounded_fitness,
            'rounded_uncertainty': rounded_uncertainty
        })

    return results


def process_completed_results(async_results, hof, temp_results_file,
                              processed_count, total_combinations,
                              progress_bar, status_text, start_time):
    """Process any completed async results"""
    # Check which results are ready
    completed_indices = []
    for i, (future, batch_size) in enumerate(async_results):
        if future.done():
            completed_indices.append(i)

            try:
                # Get the result
                result_batch = future.result()
            except Exception as e:
                st.error(f"Error in batch processing: {e}")
                continue

            # Process results and update Pareto front
            with open(temp_results_file, 'a') as f:
                for result in result_batch:
                    # Create individual
                    individual = creator.Individual(result['params'])
                    individual.fitness.values = tuple(result['predictions'])
                    individual.weighted_sum = 0.0  # to be calculated after normalization
                    individual.uncertainty = tuple(result['uncertainties'])

                    # Update Pareto front
                    hof.update([individual])

                    # Write to CSV directly to save memory
                    row_data = [str(float(p)) for p in result['params']] + result['rounded_fitness'] + result[
                        'rounded_uncertainty']
                    f.write(','.join(row_data) + '\n')

            # Update progress
            processed_count += batch_size
            progress = min(1.0, processed_count / total_combinations)
            progress_bar.progress(progress)

            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time

            status_text.text(f"Processed {processed_count}/{total_combinations} combinations. " +
                             f"Elapsed: {elapsed_time:.2f}s. Estimated remaining: {remaining_time:.2f}s.")

    # Remove completed results
    for i in sorted(completed_indices, reverse=True):
        async_results.pop(i)

    return processed_count


# Function to perform streaming optimization with optimized parallel processing
@st.cache_resource
def optimize_parameters_parallel(param_ranges, num_objectives, _models, weights, output_path, objective_names, objective_uncertainties=None, models_signature: str = "", num_cores: int = 1):
    try:
        st.write("Starting optimization process with streaming parallel processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        hof = tools.ParetoFront()

        # Get parameter combinations generator and total count
        param_generator, total_combinations = parameter_combinations_generator(param_ranges)

        st.write(f"Total parameter combinations to evaluate: {total_combinations}")

        # Threads for parallel evaluation (caps to available CPU cores)
        requested_cores = int(num_cores) if num_cores is not None else int(st.session_state.get('num_cores', 8))
        available_cores = os.cpu_count() or requested_cores
        num_cores = max(1, min(requested_cores, available_cores))
        st.write(f"Using {num_cores} threads for processing (requested: {requested_cores}, available: {available_cores})")

        # Calculate optimal batch size for Streamlit Cloud (smaller batches for stability)
        if total_combinations < 1000:
            batch_size = max(1, min(250, total_combinations // 10))
        elif total_combinations < 10000:
            batch_size = max(5, min(500, total_combinations // 20))
        else:
            batch_size = max(10, min(1000, total_combinations // 50))  # Much smaller batches for large datasets

        st.write(f"Using adaptive batch size: {batch_size}")

        # Prepare for streaming processing
        processed_count = 0
        batch_count = 0

        # Prepare column labels for the DataFrame
        param_labels = list(param_ranges.keys())
        fitness_labels = list(objective_names)
        uncertainty_labels = [f'Uncertainty_{name}' for name in objective_names]

        # Create DataFrame for results with chunked writing
        df_columns = param_labels + fitness_labels + uncertainty_labels

        # Create a temporary file for results
        temp_results_file = "temp_results.csv"
        with open(temp_results_file, 'w') as f:
            # Write header
            f.write(','.join(df_columns) + '\n')

        # Create a pool of worker threads (more Streamlit-friendly than processes)
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            # Use a queue of async results to maximize CPU utilization
            async_results = []
            batches = []
            current_batch = []

            # Process parameter combinations in streaming batches
            for i, params in enumerate(param_generator):
                current_batch.append(params)

                # When batch is full, submit it asynchronously
                if len(current_batch) >= batch_size:
                    batch_count += 1
                    batches.append(current_batch)

                    # Submit batch for async processing
                    future = executor.submit(evaluate_batch, current_batch, _models, num_objectives, objective_uncertainties)
                    async_results.append((future, len(current_batch)))

                    # Reset current batch
                    current_batch = []

                # Process completed results to free up memory
                processed_count = process_completed_results(
                    async_results, hof, temp_results_file,
                    processed_count, total_combinations,
                    progress_bar, status_text, start_time
                )

                # Periodic memory cleanup and progress check
                if i % 100 == 0:
                    import gc
                    gc.collect()
            # Submit any remaining combinations
            if current_batch:
                batch_count += 1
                batches.append(current_batch)

                # Submit batch for async processing
                future = executor.submit(evaluate_batch, current_batch, _models, num_objectives, objective_uncertainties)
                async_results.append((future, len(current_batch)))

            # Wait for all remaining results to complete
            while async_results:
                processed_count = process_completed_results(
                    async_results, hof, temp_results_file,
                    processed_count, total_combinations,
                    progress_bar, status_text, start_time
                )
                time.sleep(0.1)  # Short sleep to prevent CPU spinning

        # Complete progress bar
        progress_bar.progress(1.0)

        # Read results from temporary file
        results_df = pd.read_csv(temp_results_file, encoding='latin-1')

        # Calculate weighted sum for each solution
        if 'Weighted Sum' not in results_df.columns:
            # Normalize and calculate weighted sum
            weighted_sum = np.zeros(len(results_df))
            for i, (obj, w) in enumerate(zip(fitness_labels, weights)):
                col_values = results_df[obj].astype(float)
                v_min = col_values.min()
                v_max = col_values.max()

                if v_max == v_min:
                    normalized = np.zeros_like(col_values)
                elif w < 0:
                    normalized = w * (v_max - col_values) / (v_max - v_min)
                else:
                    normalized = w * (col_values - v_min) / (v_max - v_min)

                weighted_sum += normalized

            results_df['Weighted Sum'] = weighted_sum
            for ind, wsum in zip(hof, weighted_sum):
                ind.weighted_sum = wsum

        # Save results to Excel file if not too large
        try:
            if len(results_df) <= 1048576:  # Excel row limit
                results_df.to_excel(output_path, index=False)
                st.success(f"Results saved to {output_path}")
                # Store the output path in session state for later use
                st.session_state['output_path'] = output_path
            else:
                csv_path = output_path.replace('.xlsx', '.csv')
                results_df.to_csv(csv_path, index=False)
                st.warning(f"Results too large for Excel. Saved to {csv_path} instead.")
                # Store the CSV path in session state for later use
                st.session_state['output_path'] = csv_path
        except Exception as e:
            st.error(f"Error saving results: {e}")
            # Fallback to CSV
            try:
                csv_path = output_path.replace('.xlsx', '.csv')
                results_df.to_csv(csv_path, index=False)
                st.success(f"Results saved to {csv_path}")
                # Store the CSV path in session state for later use
                st.session_state['output_path'] = csv_path
            except:
                st.error("Failed to save results to file.")

        # Clean up temporary file
        try:
            os.remove(temp_results_file)
        except:
            pass

        # Display optimization summary
        st.write(f"Optimization completed in {time.time() - start_time:.2f} seconds")
        st.write(f"Evaluated {processed_count} parameter combinations")
        st.write(f"Found {len(hof)} solutions on the Pareto front")

        return hof

    except Exception as e:
        st.error(f"Error in optimization: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# Helper function to normalize weighted sum consistently
def normalize_weighted_sum(objective_values, weights):
    """Calculate normalized weighted sum for given objective values and weights"""
    weighted_sum = 0.0

    for i, (obj_vals, w) in enumerate(zip(objective_values.T, weights)):
        v_min = obj_vals.min()
        v_max = obj_vals.max()

        if v_max == v_min:
            normalized = np.zeros_like(obj_vals)
        elif w < 0:
            normalized = w * (v_max - obj_vals) / (v_max - v_min)
        else:
            normalized = w * (obj_vals - v_min) / (v_max - v_min)

        weighted_sum += normalized

    return weighted_sum


# Helper function to create dimension for parallel coordinates plot with non-overlapping tick labels
def create_dimension(label, values, min_val, max_val):
    """Create a dimension for parallel coordinates with clean tick labels"""
    # Let Plotly handle tick positioning automatically for better readability
    # Only specify range and let Plotly determine optimal tick positions
    if min_val == max_val:
        # Handle edge case where all values are the same
        return dict(
            range=[min_val - 0.1, max_val + 0.1],
            label=label,
            values=values
        )
    else:
        return dict(
            range=[min_val, max_val],
            label=label,
            values=values
        )


# Visualize Pareto front using Plotly for interactivity
def visualize_pareto_front(hof, objective_names_list, param_names_list, weights):
    try:
        # Ensure objective_names_list and param_names_list are lists
        objective_names_list = list(objective_names_list)
        param_names_list = list(param_names_list)

        # Extract objective values for all solutions
        objective_values = np.array([list(ind.fitness.values) for ind in hof])

        # Calculate weighted sum for each solution
        weighted_sums = np.array([ind.weighted_sum for ind in hof])

        # Create a DataFrame for the parallel coordinates plot
        data = []
        for i, ind in enumerate(hof):
            solution_data = {}
            # Add parameter values
            for j, param_name in enumerate(param_names_list):
                solution_data[param_name] = float(ind[j])

            # Add objective values
            for j, obj_name in enumerate(objective_names_list):
                solution_data[obj_name] = float(ind.fitness.values[j])

            # Add weighted sum and solution ID
            solution_data['Weighted Sum'] = float(weighted_sums[i])
            solution_data['Solution ID'] = i

            data.append(solution_data)

        df = pd.DataFrame(data)

        # Create parallel coordinates plot with Plotly
        dimensions = []

        # Add parameter dimensions with non-overlapping tick labels
        for param in param_names_list:
            param_min = df[param].min()
            param_max = df[param].max()
            dimensions.append(create_dimension(param, df[param], param_min, param_max))

        # Add objective dimensions with non-overlapping tick labels
        for obj in objective_names_list:
            obj_min = df[obj].min()
            obj_max = df[obj].max()
            dimensions.append(create_dimension(obj, df[obj], obj_min, obj_max))

        # Add weighted sum dimension at the far right
        ws_min = df['Weighted Sum'].min()
        ws_max = df['Weighted Sum'].max()
        dimensions.append(create_dimension('Weighted Sum', df['Weighted Sum'], ws_min, ws_max))

        # Create the parallel coordinates plot
        with st.expander("Optimization Result (Pareto Front)", expanded=False):
            st.write("## Solutions in Pareto Front")
            st.write("Here you find the non-dominated solutions in Pareto Front")
            fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df['Weighted Sum'],
                    colorscale='Tealrose',
                    showscale=True,
                    colorbar=dict(title='Weighted Sum', nticks=11)
                ),
                dimensions=dimensions,
                unselected=dict(line=dict(opacity=0)),
                rangefont=dict(family="Arial", size=14),
                tickfont=dict(family="Arial", size=11, weight="bold", color="black"),

            )
            )

            fig.update_layout(
                title="Solutions in Pareto Front",
                height=600,
                margin=dict(l=50, r=50, t=100, b=50),  # Increased left and right margins for axis labels
                font=dict(family='Arial', size=14)  # Reduced font size to prevent overlapping
            )

            # Display the interactive plot
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error visualizing Pareto front: {e}")
        import traceback
        st.code(traceback.format_exc())


# Visualize a single selected solution
def visualize_selected_solution(solution, param_names, objective_names, weights, hof):
    try:
        st.write("## Interactive Solution Explorer")
        st.write("Here you can isolate and investigate a specific non-dominated solutions from Pareto Front")
        st.subheader("Selected Solution Details")

        # Ensure param_names and objective_names are lists
        param_names = list(param_names)
        objective_names = list(objective_names)

        # Create tables for parameter and objective values (formatted to exactly 2 decimal places)
        col1, col2 = st.columns(2)

        with col1:
            st.write("Parameter Values")
            param_df = pd.DataFrame({
                'Parameter': param_names,
                'Value': [f"{float(p):.2f}" for p in solution]  # Format to exactly 2 decimal places
            })
            st.table(param_df)

        with col2:
            st.write("Objective Values")
            obj_df = pd.DataFrame({
                'Objective': objective_names,
                'Value': [f"{float(f):.2f}" for f in solution.fitness.values],  # Format to exactly 2 decimal places
                '(±)': [f"{float(u):.2f}" for u in solution.uncertainty],
                # Format to exactly 2 decimal places
                'Weight': [f"{float(w):.2f}" for w in weights]  # Format to exactly 2 decimal places
            })
            st.table(obj_df)

        # Create the parallel coordinates plot for the selected solution using Plotly for consistency
        # Combine parameter values and objective values for visualization
        all_names = param_names + objective_names

        # Extract all data for all solutions (parameters + objectives)
        all_data = []
        for ind in hof:
            solution_data = {}
            # Add parameter values
            for j, param_name in enumerate(param_names):
                solution_data[param_name] = float(ind[j])

            # Add objective values
            for j, obj_name in enumerate(objective_names):
                solution_data[obj_name] = float(ind.fitness.values[j])

            # Add solution ID
            solution_data['is_selected'] = 0  # Not selected

            all_data.append(solution_data)

        # Add selected solution data
        selected_data = {}
        # Add parameter values
        for j, param_name in enumerate(param_names):
            selected_data[param_name] = float(solution[j])

        # Add objective values
        for j, obj_name in enumerate(objective_names):
            selected_data[obj_name] = float(solution.fitness.values[j])

        # Mark as selected
        selected_data['is_selected'] = 1  # Selected

        # Create DataFrame with all solutions
        df_all = pd.DataFrame(all_data)
        df_selected = pd.DataFrame([selected_data])

        # Combine all data
        df_combined = pd.concat([df_all, df_selected])

        # Create dimensions for parallel coordinates with non-overlapping tick labels
        dimensions = []
        for name in all_names:
            name_min = df_combined[name].min()
            name_max = df_combined[name].max()
            dimensions.append(create_dimension(name, df_combined[name], name_min, name_max))

        # Create figure
        fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_combined['is_selected'],
                colorscale=[[0, 'rgba(250,250,250,0.5)'], [1, 'rgba(255,0,0,1)']],
                showscale=False
            ),
            dimensions=dimensions,
            unselected=dict(line=dict(opacity=0)),
            rangefont=dict(family="Arial", size=14),
            tickfont=dict(family="Arial", size=12, weight="bold", color="black")
        )
        )

        fig.update_layout(
            title="Selected Solution",
            height=600,
            margin=dict(l=50, r=50, t=100, b=50),  # Increased margins for labels
            font=dict(family="Arial", size=14)  # Smaller font to prevent overlap
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error visualizing selected solution: {e}")
        import traceback
        st.code(traceback.format_exc())


# Visualize parameter trends
def visualize_parameter_trends(hof, param_names, objective_names):
    with st.expander("Trend Analysis", expanded=False):
        try:
            st.write("## Trend Analysis")
            st.write("Here you can investigate how each parameter affects objectives")
            # Ensure param_names and objective_names are lists
            param_names = list(param_names)
            objective_names = list(objective_names)

            # Extract data for all solutions
            param_data = np.zeros((len(hof), len(param_names)))
            objective_data = np.zeros((len(hof), len(objective_names)))

            for i, ind in enumerate(hof):
                # Add parameter values
                for j, param in enumerate(ind):
                    param_data[i, j] = float(param)

                # Add objective values
                for j, obj in enumerate(ind.fitness.values):
                    objective_data[i, j] = float(obj)

            # Create a trend diagram for each parameter
            for p_idx, param_name in enumerate(param_names):
                st.subheader(f"Parameter: {param_name}")

                # Create a figure with subplots for each objective (without frame)
                fig, axes = plt.subplots(1, len(objective_names), figsize=(15, 4), frameon=False)
                if len(objective_names) == 1:
                    axes = [axes]

                for o_idx, (obj_name, ax) in enumerate(zip(objective_names, axes)):
                    # Extract parameter and objective values
                    x = param_data[:, p_idx]
                    y = objective_data[:, o_idx]

                    # Plot scatter points
                    ax.scatter(x, y, alpha=0.5, color='green')

                    # Add trendline if there are enough points
                    if len(x) > 1:
                        try:
                            # Fit a polynomial of degree 1 (linear) or 2 (quadratic) based on data size
                            degree = 2 if len(x) > 5 else 1
                            z = np.polyfit(x, y, degree)
                            p = np.poly1d(z)

                            # Generate x values for the trendline
                            x_trend = np.linspace(min(x), max(x), 100)

                            # Plot the trendline
                            ax.plot(x_trend, p(x_trend), "r--")
                        except:
                            # If fitting fails, skip trendline
                            pass

                    ax.set_xlabel(param_name)
                    ax.set_ylabel(obj_name)
                    ax.set_title(f"{param_name} vs {obj_name}")
                    ax.grid(True, linestyle='--', alpha=0.7)

                plt.tight_layout()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error visualizing parameter trends: {e}")
            import traceback
            st.code(traceback.format_exc())


# Initialize session state variables
def initialize_session_state():
    if 'models' not in st.session_state:
        st.session_state['models'] = []

    if 'optimization_complete' not in st.session_state:
        st.session_state['optimization_complete'] = False

    if 'hof' not in st.session_state:
        st.session_state['hof'] = None

    if 'param_names' not in st.session_state:
        st.session_state['param_names'] = []

    if 'objective_names' not in st.session_state:
        st.session_state['objective_names'] = []

    if 'weights' not in st.session_state:
        st.session_state['weights'] = []

    if 'selected_solution_index' not in st.session_state:
        st.session_state['selected_solution_index'] = 0

    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = "Training"

    if 'num_cores' not in st.session_state:
        st.session_state['num_cores'] = 8

    if 'output_path' not in st.session_state:
        st.session_state['output_path'] = "evaluated_solutions.xlsx"


# Callback for solution selection
def on_solution_select():
    # This function will be called when a solution is selected
    # Get the key that triggered the callback
    for key in st.session_state:
        if key.startswith('solution_selector_') and st.session_state[key] != st.session_state.get(
                'selected_solution_index', 0):
            st.session_state['selected_solution_index'] = st.session_state[key]
            break


# Function to display optimization results in a separate mode
def display_optimization_results():
    st.title("Optimization Results Explorer")
    st.write("Explore optimization results")

    # Get data from session state
    hof = st.session_state['hof']
    param_names_list = st.session_state['param_names']
    objective_names_list = st.session_state['objective_names']
    weights = st.session_state['weights']
    if not hof:
        st.error("No optimization results found.")
        return

    # Display Pareto front visualization
    visualize_pareto_front(hof, objective_names_list, param_names_list, weights)

    # Display Parameter Trend Analysis (only once)
    visualize_parameter_trends(hof, param_names_list, objective_names_list)

    # Recalculate normalized weighted sum for consistent display
    objective_values = np.array([list(ind.fitness.values) for ind in hof])
    normalized_weighted_sums = normalize_weighted_sum(objective_values, weights)

    # Update the weighted sum in each individual
    for ind, norm_wsum in zip(hof, normalized_weighted_sums):
        ind.weighted_sum = norm_wsum

    # Sort solutions by normalized weighted sum (lowest to highest)
    sorted_solutions = sorted(range(len(hof)), key=lambda i: hof[i].weighted_sum, reverse=False)

    # Create a mapping from sorted index to original index
    sorted_to_original = {i: sorted_idx for i, sorted_idx in enumerate(sorted_solutions)}
    st.session_state['sorted_to_original'] = sorted_to_original

    # Interactive solution explorer
    with st.expander("Interactive Solution Explorer", expanded=False):
        st.write("Here you can isolate and investigate a specific non-dominated solutions from Pareto Front")

        # Create a unique key for the solution selector to avoid duplicate key errors
        # Use a timestamp to ensure uniqueness across reruns
        if 'explorer_key_suffix' not in st.session_state:
            st.session_state['explorer_key_suffix'] = str(int(time.time()))

        unique_selector_key = f"solution_selector_{st.session_state['explorer_key_suffix']}"

        # Create selectbox with sorted solutions using callback
        selected_solution_sorted = st.selectbox(
            "Select solution to explore:",
            options=range(len(sorted_solutions)),
            format_func=lambda i: f"Solution {i + 1} (Weighted Sum: {hof[sorted_to_original[i]].weighted_sum:.2f})",
            key=unique_selector_key,
            index=st.session_state['selected_solution_index'],
            on_change=on_solution_select
        )

        # Get the original index
        selected_solution_original = sorted_to_original[selected_solution_sorted]

        # Visualize the selected solution
        solution = hof[selected_solution_original]
        visualize_selected_solution(solution, param_names_list, objective_names_list, weights, hof)

    # === NEW TAB: Estimate Objectives for Custom Parameter Combination ===
    with st.expander("Estimate Objectives Results for Custom Parameters Values", expanded=False):
        st.write("## Estimate Objectives Results for Custom Parameters Values")
        st.write(
            "Here you can estimate what is expected to be the result for each individual objective for entered parameter values")
        st.write("### Enter parameter values to simulate:")

        input_values = {}
        col1, col2, col3 = st.columns(3)
        for i, param in enumerate(param_names_list):
            col = [col1, col2, col3][i % 3]
            with col:
                input_values[param] = st.number_input(f"{param}", key=f"custom_input_{i}")

        input_array = np.array([[input_values[p] for p in param_names_list]])

        if st.button("Estimate Objectives Results"):
            models = st.session_state.get("models", [])
            if not models:
                st.error("No trained models found.")
            else:
                st.subheader("Estimated Objective Values")
                preds = [model.predict(input_array)[0] for model in models]
                df_estimates = pd.DataFrame({
                    "Objective": objective_names_list,
                    "Estimated Value": [f"{p:.2f}" for p in preds]
                })

                # Create diagrams for each objective with color scale
                for i, (obj, pred) in enumerate(zip(objective_names_list, preds)):
                    obj_values = [float(ind.fitness.values[i]) for ind in hof]
                    obj_min, obj_max = min(obj_values), max(obj_values)
                    color_value = (pred - obj_min) / (obj_max - obj_min) if obj_max > obj_min else 0.5

                all_pred_values = []
                all_obj_names = []
                all_obj_min = []
                all_obj_max = []
                all_normalized_values = []
                all_percentiles = []
                all_percentiles = []

                # Collect values from the loop and normalize each objective individually
                for i, (obj, pred) in enumerate(zip(objective_names_list, preds)):
                    obj_values = [float(ind.fitness.values[i]) for ind in hof]
                    obj_min, obj_max = min(obj_values), max(obj_values)

                    # Normalize the prediction value to 0-100 scale for this specific objective
                    if obj_max > obj_min:
                        normalized_pred = ((float(pred) - obj_min) / (obj_max - obj_min)) * 100
                    else:
                        normalized_pred = 50  # If min == max, put it in the middle

                    all_pred_values.append(float(pred))  # Keep original values for hover/display
                    all_obj_names.append(str(obj))
                    all_obj_min.append(obj_min)
                    all_obj_max.append(obj_max)
                    try:
                        _vals = np.asarray(obj_values, dtype=float)
                        percentile = float(np.mean(_vals <= float(pred)) * 100.0) if len(_vals) else 50.0
                    except Exception:
                        percentile = 50.0
                    all_normalized_values.append(normalized_pred)
                    all_percentiles.append(percentile)

                # CLOSE THE POLYGON: Add the first point at the end
                all_normalized_values_closed = all_normalized_values + [all_normalized_values[0]]
                all_obj_names_closed = all_obj_names + [all_obj_names[0]]
                all_pred_values_closed = all_pred_values + [all_pred_values[0]]
                all_obj_min_closed = all_obj_min + [all_obj_min[0]]
                all_obj_max_closed = all_obj_max + [all_obj_max[0]]

                # Create single radar chart with individual axis scaling and closed polygon

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=all_normalized_values_closed,  # Use closed data
                    theta=all_obj_names_closed,  # Use closed data
                    fill='toself',
                    fillcolor='rgba(154, 205, 50, 0.3)',  # yellowgreen with transparency
                    line=dict(color='yellowgreen', width=4),
                    marker=dict(size=8, color='yellowgreen'),
                    name='Predictions',
                    text=[f'{obj}: {val:.2f}<br>Range: {min_val:.2f} - {max_val:.2f}'
                          for obj, val, min_val, max_val in
                          zip(all_obj_names_closed, all_pred_values_closed, all_obj_min_closed, all_obj_max_closed)],
                    hovertemplate='%{text}<br>Normalized: %{r:.1f}%<extra></extra>'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],  # Fixed 0-100 scale since we normalized the values
                            ticksuffix='%',
                            gridcolor='lightslategrey',
                            tickmode='linear',
                            tick0=0,
                            dtick=20  # Show ticks every 20%
                        ),
                        angularaxis=dict(
                            gridcolor='lightslategrey'
                        )
                    ),
                    height=500,
                    margin=dict(t=50, b=50),
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display the values in a table for reference with individual scales (use original data, not closed)
                df_estimates_detailed = pd.DataFrame({
                    "Objective": all_obj_names,  # Use original data for table
                    "Estimated Value": [f"{p:.2f}" for p in all_pred_values],
                    "Min Range": [f"{m:.2f}" for m in all_obj_min],
                    "Max Range": [f"{m:.2f}" for m in all_obj_max],
                    "Normalized (%)": [f"{n:.1f}%" for n in all_normalized_values],
                    "Percentile": [f"{p:.1f}%" for p in all_percentiles],
                })
                st.table(df_estimates_detailed)

    # === NEW TAB: All Computed Solutions ===
    with st.expander("All Computed Solutions (Interactive)", expanded=False):
        st.write("## All Computed Solutions (Interactive)")
        st.write(
            "Here you can filter solutions by parameter and|or objective values range for a deeper investigation of results")
        try:
            # IMPROVEMENT 1: Automatically point to the generated Excel file
            # Use the output path from session state if available
            default_path = st.session_state.get(
                "explorer_uploaded_file_path") if "explorer_uploaded_file_path" in st.session_state else st.session_state.get(
                "output_path", "evaluated_solutions.xlsx")
            output_path = st.text_input("Path to Results File", value=default_path)

            # Load data
            if output_path.endswith(".xlsx") and os.path.exists(output_path):
                df_all = pd.read_excel(output_path, engine='openpyxl')
            elif output_path.endswith(".csv") and os.path.exists(output_path):
                df_all = pd.read_csv(output_path)
            else:
                st.warning("Results file not found.")
                df_all = None

            if df_all is not None:
                # Calculate weighted sum if not present
                if 'Weighted Sum' not in df_all.columns:
                    # Calculate weighted sum based on objectives and weights
                    weighted_sum = np.zeros(len(df_all))
                    for i, (obj, w) in enumerate(zip(objective_names_list, weights)):
                        if obj in df_all.columns:
                            weighted_sum += df_all[obj].astype(float) * w

                    # Add weighted sum to the DataFrame
                    df_all['Weighted Sum'] = weighted_sum

                # Add filters for each parameter
                st.write("### Filter by Parameter Values")

                # Add filters for each parameter
                filters = {}
                for col in st.session_state['param_names']:
                    min_val = float(df_all[col].min())
                    max_val = float(df_all[col].max())
                    selected_range = st.slider(
                        f"{col}", min_val, max_val, (max_val, min_val), step=0.1
                    )
                    filters[col] = selected_range

                st.write("### Filter by Objective Values")
                for col in st.session_state['objective_names']:
                    min_val = float(df_all[col].min())
                    max_val = float(df_all[col].max())
                    selected_range = st.slider(
                        f"{col}", min_val, max_val, (min_val, max_val), step=0.1
                    )
                    filters[col] = selected_range

                # Weighted Sum filter
                st.write("### Filter by Weighted Sum")
                min_val = float(df_all['Weighted Sum'].min())
                max_val = float(df_all['Weighted Sum'].max())
                selected_range = st.slider(
                    "Weighted Sum", min_val, max_val, (min_val, max_val), step=0.1
                )
                filters['Weighted Sum'] = selected_range

                # Apply all filters
                filtered_df = df_all.copy()
                for col, (low, high) in filters.items():
                    filtered_df = filtered_df[(filtered_df[col] >= low) & (filtered_df[col] <= high)]

                st.write(f"Filtered to {len(filtered_df)} solutions.")

                # Create dimensions for parallel coordinates with non-overlapping tick labels
                dimensions_all = []

                combined_cols = st.session_state['param_names'] + st.session_state['objective_names']
                if 'Weighted Sum' in combined_cols:
                    combined_cols.remove('Weighted Sum')

                for col in combined_cols:
                    col_min = filtered_df[col].min()
                    col_max = filtered_df[col].max()
                    dimensions_all.append(create_dimension(col, filtered_df[col], col_min, col_max))

                # Add weighted sum axis at the far right
                weighted_min = filtered_df['Weighted Sum'].min()
                weighted_max = filtered_df['Weighted Sum'].max()
                dimensions_all.append(
                    create_dimension('Weighted Sum', filtered_df['Weighted Sum'], weighted_min, weighted_max))

                # IMPROVEMENT 3: Change unselected solutions to light gray
                # Ensure we use Weighted Sum for color scaling
                fig_all = go.Figure(data=go.Parcoords(
                    line=dict(
                        color=filtered_df["Weighted Sum"],  # Always use Weighted Sum for color
                        colorscale='Tealrose',
                        showscale=True,
                        colorbar=dict(title='Weighted Sum', nticks=11)
                    ),
                    dimensions=dimensions_all,
                    unselected=dict(line=dict(opacity=0)),
                    rangefont=dict(family="Arial", size=14),
                    tickfont=dict(family="Arial", size=11, weight="bold", color="black")
                ))

                fig_all.update_layout(
                    title="All Computed Solutions (Filtered)",
                    height=600,
                    margin=dict(l=50, r=50, t=100, b=50),
                    font=dict(family="Arial", size=14)
                )

                st.plotly_chart(fig_all, use_container_width=True)

                # Export button for filtered solutions
                if len(filtered_df) > 0:
                    # Convert DataFrame to Excel bytes for download
                    from io import BytesIO
                    excel_buffer = BytesIO()
                    filtered_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                    excel_data = excel_buffer.getvalue()

                    st.download_button(
                        label="Export Filtered Solutions",
                        data=excel_data,
                        file_name="filtered_solutions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Error in All Computed Solutions section: {e}")
            import traceback
            st.code(traceback.format_exc())


# Helper function to calculate weighted sum for Explorer tab
def calculate_explorer_weighted_sum(objective_values, weights):
    """Calculate normalized weighted sum for Explorer tab using a different scoring system"""
    try:
        # objective_values is a 2D array: (solutions x objectives)
        num_solutions, num_objectives = objective_values.shape

        # Initialize normalized scores matrix
        normalized_scores = np.zeros((num_solutions, num_objectives))

        # For each objective
        for obj_idx, weight in enumerate(weights):
            obj_values = objective_values[:, obj_idx]

            # Find min and max values for this objective
            min_val = np.min(obj_values)
            max_val = np.max(obj_values)

            if max_val == min_val:
                # All values are the same, give everyone the full weight
                normalized_scores[:, obj_idx] = abs(weight)
            else:
                # Determine if we're minimizing or maximizing based on weight sign
                if weight < 0:  # Minimizing objective
                    # Best (minimum) value gets full weight, worst (maximum) gets 0
                    normalized_scores[:, obj_idx] = abs(weight) * (max_val - obj_values) / (max_val - min_val)
                else:  # Maximizing objective
                    # Best (maximum) value gets full weight, worst (minimum) gets 0
                    normalized_scores[:, obj_idx] = abs(weight) * (obj_values - min_val) / (max_val - min_val)

        # Sum all objective scores to get final weighted sum
        weighted_sums = np.sum(normalized_scores, axis=1)

        return weighted_sums

    except Exception as e:
        st.error(f"Error calculating explorer weighted sum: {e}")
        return np.zeros(len(objective_values))


# Main function
def explore_uploaded_results():
    st.title("Explorer")
    st.write("Upload and explore previously computed optimization results")

    num_objectives = st.number_input(
        "Enter Number of Objectives",
        min_value=1,
        value=4,
        key="explorer_num_objectives"
    )
    uploaded_file = st.file_uploader(
        "Upload Optimization Results (.xlsx or .csv)",
        type=["xlsx", "csv"],
        key="explorer_file_upload"
    )

    if uploaded_file is None:
        st.warning("Please upload a valid optimization result file to proceed.")
        return

    # Load the full DataFrame from uploaded file
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading uploaded file: {e}")
        return

    # Identify the 'Weighted Sum' column (must exist or will be calculated)
    ws_col = 'Weighted Sum' if 'Weighted Sum' in df.columns else None

    # Identify any uncertainty columns
    uncertainty_cols = [c for c in df.columns if c.startswith("Uncertainty_")]

    # Core data columns: drop uncertainties + weighted sum if present
    exclude_cols = uncertainty_cols + ([ws_col] if ws_col else [])
    core_cols = [c for c in df.columns if c not in exclude_cols]

    if len(core_cols) < num_objectives:
        st.error(f"Number of objectives ({num_objectives}) exceeds available data columns ({len(core_cols)}).")
        return

    # Split core_cols into parameters vs. objectives
    param_names = core_cols[:-num_objectives]
    objective_names = core_cols[-num_objectives:]
    weights = [-1.0 for _ in objective_names]  # Default to minimization

    # Allow user to adjust weights
    with st.expander("Setup Objectives Weights", expanded=False):
        st.write("### Setup Objectives Weights")
        st.write("Set weights for each objective (negative for minimization, positive for maximization)")

        col1, col2, col3 = st.columns(3)
        for i, obj_name in enumerate(objective_names):
            col = [col1, col2, col3][i % 3]
            with col:
                weights[i] = st.number_input(
                    f"Weight for {obj_name}",
                    value=-1.0,
                    help="Negative for minimization, positive for maximization",
                    key=f"explorer_weight_{obj_name}"
                )

    # Display basic statistics
    with st.expander("Dataset Overview", expanded=False):
        st.write("### Dataset Overview")
        st.write(f"**Total Solutions:** {len(df)}")
        st.write(f"**Parameters:** {', '.join(param_names)}")
        st.write(f"**Objectives:** {', '.join(objective_names)}")

    # === Estimate Objectives for Custom Parameter Combination in Explorer ===
    with st.expander("Estimate Objectives Results for Custom Parameters Values", expanded=False):
        st.write("## Estimate Objectives Results for Custom Parameters Values")
        st.write(
            "Here you can estimate what is expected to be the result for each individual objective for entered parameter values")

        st.markdown("#### Model selection (Explorer)")
        explorer_model_strategy = st.radio(
            "Model strategy for Explorer estimation",
            options=["Auto (recommended)", "Manual per objective"],
            index=0,
            key="explorer_model_strategy",
            help="Auto: PAMO selects Ridge / Random Forest / Hybrid per objective using cross-validated NMAE on the uploaded dataset."
        )

        explorer_objective_overrides = {obj: "Auto" for obj in objective_names}
        if str(explorer_model_strategy).lower().startswith("manual"):
            st.caption("Manual overrides apply only to Explorer estimation.")
            for obj in objective_names:
                explorer_objective_overrides[obj] = st.selectbox(
                    f"Model for {obj}",
                    options=["Auto", "Ridge", "Random Forest", "Hybrid"],
                    index=0,
                    key=f"explorer_model_override_{obj}"
                )

        st.write("### Enter parameter values to simulate:")

        input_values = {}
        col1, col2, col3 = st.columns(3)
        for i, param in enumerate(param_names):
            col = [col1, col2, col3][i % 3]
            with col:
                input_values[param] = st.number_input(f"{param}", key=f"explorer_custom_input_{i}")

        input_array = np.array([[input_values[p] for p in param_names]])
        if st.button("Estimate Objectives Results", key="explorer_estimate_btn"):
            try:
                # Train (or reuse) Explorer models on the uploaded dataset (do not overwrite Training-tab models)
                ridge_alpha = float(st.session_state.get("ridge_alpha", 0.1))
                cv_folds = int(st.session_state.get("cv_folds", 5))
                rf_trees = int(st.session_state.get("rf_n_estimators", 50))
                num_cores = int(st.session_state.get("num_cores", 1))

                # Use only parameter + objective columns for training
                df_train = df[param_names + objective_names].copy()

                signature = (
                    getattr(uploaded_file, "name", "uploaded"),
                    tuple(df_train.shape),
                    tuple(param_names),
                    tuple(objective_names),
                    str(explorer_model_strategy),
                    tuple(explorer_objective_overrides.get(o, "Auto") for o in objective_names),
                    float(ridge_alpha),
                    int(cv_folds),
                    int(rf_trees),
                    int(num_cores),
                )

                if st.session_state.get("explorer_models_signature") != signature:
                    st.info("Training surrogate models for Explorer estimation (per objective)...")
                    models, _, _ = preprocess_and_train_with_cv(
                        df_train,
                        len(objective_names),
                        ridge_alpha=ridge_alpha,
                        cv_folds=cv_folds,
                        model_selection_mode="Auto" if str(explorer_model_strategy).lower().startswith("auto") else "Manual",
                        objective_model_overrides=explorer_objective_overrides,
                        store_in_session=False
                    ,
                        rf_n_estimators=int(rf_trees),
                        num_cores=int(num_cores),
                        data_signature=compute_df_signature(df_train)
                    )
                    st.session_state["explorer_models"] = models
                    st.session_state["explorer_models_signature"] = signature
                else:
                    models = st.session_state.get("explorer_models", [])

                if not models:
                    st.error("No models available for Explorer estimation.")
                    st.stop()

                # Show which model was selected per objective (transparency)
                try:
                    kinds = [getattr(m, "kind", type(m).__name__) for m in models]
                    st.dataframe(pd.DataFrame({"Objective": objective_names, "Selected model": kinds}), use_container_width=True)
                except Exception:
                    pass
                st.subheader("Estimated Objective Values")
                preds = [model.predict(input_array)[0] for model in models]
                df_estimates = pd.DataFrame({
                    "Objective": objective_names,
                    "Estimated Value": [f"{p:.2f}" for p in preds]
                })
                st.dataframe(df_estimates, use_container_width=True)

                # RADAR CHART IMPLEMENTATION - REPLACES GAUGE CHARTS
                # Collect all data for radar chart
                all_pred_values = []
                all_obj_names = []
                all_obj_min = []
                all_obj_max = []
                all_normalized_values = []
                all_percentiles = []

                # Collect values from the loop and normalize each objective individually
                for i, (obj, pred) in enumerate(zip(objective_names, preds)):
                    obj_values = df[obj].values
                    obj_min, obj_max = obj_values.min(), obj_values.max()

                    # Normalize the prediction value to 0-100 scale for this specific objective
                    if obj_max > obj_min:
                        normalized_pred = ((float(pred) - obj_min) / (obj_max - obj_min)) * 100
                    else:
                        normalized_pred = 50  # If min == max, put it in the middle

                    try:
                        _vals = np.asarray(obj_values, dtype=float)
                        percentile = float(np.mean(_vals <= float(pred)) * 100.0) if len(_vals) else 50.0
                    except Exception:
                        percentile = 50.0

                    all_pred_values.append(float(pred))  # Keep original values for hover/display
                    all_obj_names.append(str(obj))
                    all_obj_min.append(obj_min)
                    all_obj_max.append(obj_max)
                    all_normalized_values.append(normalized_pred)
                    all_percentiles.append(percentile)

                # Close the polygon: Add the first point at the end
                all_normalized_values_closed = all_normalized_values + [all_normalized_values[0]]
                all_obj_names_closed = all_obj_names + [all_obj_names[0]]
                all_pred_values_closed = all_pred_values + [all_pred_values[0]]
                all_obj_min_closed = all_obj_min + [all_obj_min[0]]
                all_obj_max_closed = all_obj_max + [all_obj_max[0]]

                # Create single radar chart with individual axis scaling and closed polygon
                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=all_normalized_values_closed,  # Use closed data
                    theta=all_obj_names_closed,  # Use closed data
                    fill='toself',
                    fillcolor='rgba(154, 205, 50, 0.3)',  # yellowgreen with transparency
                    line=dict(color='yellowgreen', width=4),
                    marker=dict(size=8, color='yellowgreen'),
                    name='Predictions',
                    text=[f'{obj}: {val:.2f}<br>Range: {min_val:.2f} - {max_val:.2f}'
                          for obj, val, min_val, max_val in
                          zip(all_obj_names_closed, all_pred_values_closed, all_obj_min_closed, all_obj_max_closed)],
                    hovertemplate='%{text}<br>Normalized: %{r:.1f}%<extra></extra>'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],  # Fixed 0-100 scale since we normalized the values
                            ticksuffix='%',
                            gridcolor='lightslategrey',
                            tickmode='linear',
                            tick0=0,
                            dtick=20  # Show ticks every 20%
                        ),
                        angularaxis=dict(
                            gridcolor='lightslategrey'
                        )
                    ),
                    height=500,
                    margin=dict(t=50, b=50),
                    showlegend=False,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display the values in a table for reference with individual scales (use original data, not closed)
                df_estimates_detailed = pd.DataFrame({
                    "Objective": all_obj_names,  # Use original data for table
                    "Estimated Value": [f"{p:.2f}" for p in all_pred_values],
                    "Min Range": [f"{m:.2f}" for m in all_obj_min],
                    "Max Range": [f"{m:.2f}" for m in all_obj_max],
                    "Normalized (%)": [f"{n:.1f}%" for n in all_normalized_values],
                    "Percentile": [f"{p:.1f}%" for p in all_percentiles],
                })
                st.table(df_estimates_detailed)

            except Exception as e:
                st.error(f"Error training models or making predictions: {e}")

    # Display the All Computed Solutions diagram using uploaded data with new weighted sum calculation
    with st.expander("All Computed Solutions (Interactive)", expanded=False):
        st.write("## All Computed Solutions (Interactive) - Explorer")
        st.write(
            "Here you can filter solutions by parameter and|or objective values range for a deeper investigation of uploaded results")

        try:
            # Calculate weighted sum using the new normalized scoring system for Explorer only
            df_display = df.copy()
            if 'Weighted Sum' not in df_display.columns:
                # Extract objective values for calculation
                objective_values = []
                for obj in objective_names:
                    if obj in df_display.columns:
                        objective_values.append(df_display[obj].astype(float).values)

                if objective_values:
                    # Convert to numpy array (solutions x objectives)
                    objective_matrix = np.column_stack(objective_values)
                    # Calculate normalized weighted sum using the new scoring system
                    weighted_sums = calculate_explorer_weighted_sum(objective_matrix, weights)
                    # Add weighted sum to the DataFrame
                    df_display['Weighted Sum'] = weighted_sums

            # Add filters for each parameter
            st.write("### Filter by Parameter Values")
            filters = {}
            for col in param_names:
                if col in df_display.columns:
                    min_val = float(df_display[col].min())
                    max_val = float(df_display[col].max())
                    selected_range = st.slider(
                        f"{col}", min_val, max_val, (min_val, max_val), step=0.1,
                        key=f"explorer_param_{col}"
                    )
                    filters[col] = selected_range

            st.write("### Filter by Objective Values")
            for col in objective_names:
                if col in df_display.columns:
                    min_val = float(df_display[col].min())
                    max_val = float(df_display[col].max())
                    selected_range = st.slider(
                        f"{col}", min_val, max_val, (min_val, max_val), step=0.1,
                        key=f"explorer_obj_{col}"
                    )
                    filters[col] = selected_range

            # Weighted Sum filter
            st.write("### Filter by Weighted Sum")
            min_val = float(df_display['Weighted Sum'].min())
            max_val = float(df_display['Weighted Sum'].max())
            selected_range = st.slider(
                "Weighted Sum", min_val, max_val, (min_val, max_val), step=0.1,
                key="explorer_weighted_sum"
            )
            filters['Weighted Sum'] = selected_range

            # Apply all filters
            filtered_df = df_display.copy()
            for col, (low, high) in filters.items():
                if col in filtered_df.columns:
                    filtered_df = filtered_df[(filtered_df[col] >= low) & (filtered_df[col] <= high)]

            st.write(f"Filtered to {len(filtered_df)} solutions from uploaded data.")

            # Create dimensions for parallel coordinates
            dimensions_all = []

            # Add parameter dimensions
            for col in param_names:
                if col in filtered_df.columns:
                    col_min = filtered_df[col].min()
                    col_max = filtered_df[col].max()
                    dimensions_all.append(dict(
                        range=[col_min, col_max],
                        label=col,
                        values=filtered_df[col]
                    ))

            # Add objective dimensions
            for col in objective_names:
                if col in filtered_df.columns:
                    col_min = filtered_df[col].min()
                    col_max = filtered_df[col].max()
                    dimensions_all.append(dict(
                        range=[col_min, col_max],
                        label=col,
                        values=filtered_df[col]
                    ))

            # Add weighted sum axis at the far right
            if 'Weighted Sum' in filtered_df.columns:
                weighted_min = filtered_df['Weighted Sum'].min()
                weighted_max = filtered_df['Weighted Sum'].max()
                dimensions_all.append(dict(
                    range=[weighted_max, weighted_min],
                    label='Score',
                    values=filtered_df['Weighted Sum']
                ))

            # Create parallel coordinates plot for uploaded data with EXACT same parameters as original
            fig_all = go.Figure(data=go.Parcoords(
                line=dict(
                    color=filtered_df["Weighted Sum"] if 'Weighted Sum' in filtered_df.columns else filtered_df.iloc[:,
                                                                                                    -1],
                    colorscale='Tealrose_r',
                    showscale=True,
                    colorbar=dict(title='Score', nticks=11)
                ),
                dimensions=dimensions_all,
                unselected=dict(line=dict(opacity=0)),
                rangefont=dict(family="Arial", size=14),
                tickfont=dict(family="Arial", size=11, weight="bold", color="black")
            ))

            fig_all.update_layout(
                title="All Computed Solutions (Filtered)",
                height=600,
                margin=dict(l=50, r=50, t=100, b=50),
                font=dict(family="Arial", size=14)
            )

            st.plotly_chart(fig_all, use_container_width=True)

            # Export button for filtered solutions from uploaded data
            if len(filtered_df) > 0:
                # Convert DataFrame to Excel bytes for download
                from io import BytesIO
                excel_buffer = BytesIO()
                filtered_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                excel_data = excel_buffer.getvalue()

                st.download_button(
                    label="Export Filtered Solutions",
                    data=excel_data,
                    file_name="uploaded_filtered_solutions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="explorer_export_btn",
                    use_container_width=True
                )

        except Exception as e:
            st.error(f"Error in All Computed Solutions section: {e}")
            import traceback
            st.code(traceback.format_exc())


# Main function
def main():
    try:
        # Initialize session state
        initialize_session_state()

        # Sidebar for configuration
        with st.sidebar.expander("Setup", expanded=False):
            st.header("Setup")
            ridge_alpha = st.slider("Ridge alpha (regularization)", min_value=0.001, max_value=10.0, value=0.3, step=0.01,
                                     format="%.2f", help="Higher alpha = more regularization (smoother model).")
            st.session_state['ridge_alpha'] = float(ridge_alpha)

            rf_n_estimators = st.slider(
                "Number of Trees in Random Forest",
                min_value=1, max_value=200, value=int(st.session_state.get('rf_n_estimators', 50)), step=1,
                help="Controls Random Forest complexity/smoothness. More trees improve stability but increase runtime."
            )
            st.session_state['rf_n_estimators'] = int(rf_n_estimators)

            model_selection_mode = st.radio(
                "Model strategy",
                options=["Auto (recommended)", "Manual per objective"],
                index=0,
                help="Auto: PAMO selects Ridge / Random Forest / Hybrid independently per objective using cross-validated NMAE.\nManual: you can override the choice per objective in the Training tab."
            )
            st.session_state['model_selection_mode'] = model_selection_mode

            cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=20, value=5,
                                 help="Only change if you are sure what you are doing")
            st.session_state['cv_folds'] = int(cv_folds)
            st.session_state['num_cores'] = st.slider("Number of CPU Cores", min_value=1, max_value=32, value=16,
                                                      help="Only change if you are sure what you are doing")

        # Create tabs for Training, Results and Explorer
        tab1, tab2, tab3 = st.tabs(["Training", "Results", "Explorer"])

        # Results tab content
        with tab2:
            if st.session_state['optimization_complete'] and st.session_state['hof']:
                display_optimization_results()
            else:
                st.info("Run optimization in the Training tab to see results here.")
                if st.button("Go to Training Tab"):
                    st.session_state['active_tab'] = "Training"
                    st.rerun()

        # Explorer tab content
        with tab3:
            explore_uploaded_results()

        # Training tab content
        with tab1:
            st.title("Training and Optimization")
            st.write("Setup and Train Machine Learning Model")
            # Normal parameter setup flow
            num_objectives = st.number_input("Enter Number of Objectives", min_value=1, value=5)
            df = load_data(num_objectives)
            if df is None:
                return

            # Verify that the number of objectives matches the data
            if len(df.columns) > 0:
                actual_objectives = min(num_objectives, len(df.columns))
                if actual_objectives != num_objectives:
                    st.warning(
                        f"Number of objectives ({num_objectives}) adjusted to match data columns ({actual_objectives}).")
                    num_objectives = actual_objectives

            # Use the enhanced training function
            with st.expander("Model Training and Validation", expanded=False):
                st.header("Model Training and Validation")
                st.subheader("Cross-Validation Results")
                X = df.drop(df.columns[-num_objectives:], axis=1)
                y = [df[col] for col in df.columns[-num_objectives:]]
                objective_names = df.columns[-num_objectives:].tolist()

                # Manual overrides (optional)
                model_selection_mode = st.session_state.get('model_selection_mode', 'Auto (recommended)')
                objective_model_overrides = {}

                if str(model_selection_mode).lower().startswith('manual'):
                    st.subheader("Manual model selection per objective")
                    st.caption("If set to Auto, PAMO will decide based on cross-validated NMAE.")
                    for obj in objective_names:
                        objective_model_overrides[obj] = st.selectbox(
                            f"Model for {obj}",
                            options=["Auto", "Ridge", "Random Forest", "Hybrid"],
                            index=0,
                            key=f"model_override_{obj}"
                        )
                else:
                    objective_model_overrides = {obj: "Auto" for obj in objective_names}

                st.session_state['objective_model_overrides'] = objective_model_overrides

                models, param_names, objective_names = preprocess_and_train_with_cv(
                    df,
                    num_objectives,
                    ridge_alpha=ridge_alpha,
                    cv_folds=cv_folds,
                    model_selection_mode="Auto" if str(model_selection_mode).lower().startswith("auto") else "Manual",
                    objective_model_overrides=objective_model_overrides,
                    rf_n_estimators=int(st.session_state.get('rf_n_estimators', 50)),
                    num_cores=int(st.session_state.get('num_cores', 8)),
                    data_signature=compute_df_signature(df),
                )

                if models and y:
                    st.subheader("Learning Curves")
                    objective_names_list = list(objective_names)
                    visualize_model_performance(X, y, st.session_state.get('plot_estimators', models), objective_names_list, cv_folds)

            # Define parameter ranges
            with st.expander("Setup Training and run Optimization", expanded=False):
                st.header("Setup Training and run Optimization")
                st.subheader("Define Parameters Ranges")
                param_ranges = {}

                # Create columns for better layout
                col1, col2, col3 = st.columns(3)

                # For each parameter
                for i, param in enumerate(param_names):
                    # Determine which column to use
                    col = [col1, col2, col3][i % 3]

                    with col:
                        st.write(f"**{param}**")
                        # Initialize min and max values from the training data
                        min_val = st.number_input(f"Min", value=float(X[param].min()), key=f"min_{i}")
                        max_val = st.number_input(f"Max", value=float(X[param].max()), key=f"max_{i}")

                        # Calculate a reasonable default step size (1/10 of the range)
                        default_step = max(0.1, (float(X[param].max()) - float(X[param].min())) / 10)
                        step = st.number_input(f"Step", value=default_step, key=f"step_{i}")
                        param_ranges[param] = (min_val, max_val, step)

                # Define objective weights
                st.subheader("Define Objective Weights")
                weights = []

                # Create columns for better layout
                col1, col2, col3 = st.columns(3)

                # For each objective
                for i, obj_name in enumerate(objective_names):
                    # Determine which column to use
                    col = [col1, col2, col3][i % 3]

                    with col:
                        st.write(f"**{obj_name}**")
                        tooltip = "Negative for minimization, positive for maximization"
                        weight = st.number_input(f"Weight", value=-1.0, help=tooltip, key=f"weight_{i}")
                        weights.append(weight)

                st.subheader("Run optimization")
                output_path = st.text_input("Enter the file path for the output Excel file",
                                            value="evaluated_solutions.xlsx")

                # Number of solutions to display
                # Optimize parameters
                if st.button("Optimize Parameters"):
                    # Create individual class with uncertainty storage
                    create_individual_class(num_objectives, weights)

                    # Store weights in session state
                    st.session_state['weights'] = weights
                    # Check if we have models
                    models = st.session_state.get('models', [])
                    if not models:
                        st.error("No trained models available. Please train models first.")
                        return

                    # Run optimization with parallel processing
                    with st.spinner("Optimizing parameters..."):
                        objective_names_list = list(objective_names)
                        hof = optimize_parameters_parallel(
                            param_ranges,
                            num_objectives,
                            models,
                            weights,
                            output_path,
                            objective_names_list,
                            st.session_state.get('objective_uncertainties', None),
                            models_signature=st.session_state.get('models_signature', ''),
                            num_cores=int(st.session_state.get('num_cores', 8)),
                        )

                    if hof:
                        # Store results in session state for exploration mode
                        st.session_state['hof'] = hof
                        st.session_state['param_names'] = list(param_names)
                        st.session_state['objective_names'] = list(objective_names)
                        st.session_state['optimization_complete'] = True

                        # Switch to results tab
                        st.session_state['active_tab'] = "Results"
                        st.rerun()
                    else:
                        st.warning("No solutions found or optimization failed.")

    except Exception as e:
        st.error(f"An error occurred in the main function: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

with st.sidebar:
    st.markdown("---")
    st.caption("*A product of*")
    st.image("WS_Logo.png", width=300)
    st.caption("Werner Sobek Green Technologies GmbH")
    st.caption("Fachgruppe Simulation")
    st.markdown("---")
    st.caption("*Coded by*")
    st.caption("Rodrigo Carvalho")
    st.caption("*Need help? Contact me under:*")
    st.caption("*email:* rodrigo.carvalho@wernersobek.com")
    st.caption("*Tel* +49.40.6963863-14")
    st.caption("*Mob* +49.171.964.7850")
