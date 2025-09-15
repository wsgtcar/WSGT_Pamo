
import os
import time
import itertools
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool  # threads fallback
from pickle import PicklingError

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go

from deap import base, creator, tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, learning_curve

# ========================= Streamlit Setup =========================
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="WSGT_PAMO Machine Learning",
    page_icon="Pamo_Icon_White.png",
    layout="wide",
)

st.sidebar.image("Pamo_Icon_Black.png", width=80)
st.sidebar.write("## WSGT_PAMO")
st.sidebar.write("Version 1.0.0")
st.sidebar.markdown("---")

_, col2 = st.columns([1,2])
with col2:
    st.image("WS_Logo.png", width=900)

st.title("Machine Learning Assisted Optimization")

# ========================= Worker Globals =========================
_GLOBAL_MODELS = None
_GLOBAL_NUM_OBJECTIVES = None

def _init_worker(models, num_objectives):
    """Initializer for worker processes/threads to hold models in globals."""
    global _GLOBAL_MODELS, _GLOBAL_NUM_OBJECTIVES
    _GLOBAL_MODELS = models
    _GLOBAL_NUM_OBJECTIVES = num_objectives

def _evaluate_batch_worker(batch_params):
    """Evaluate a batch of parameter tuples using global models (set by _init_worker)."""
    global _GLOBAL_MODELS, _GLOBAL_NUM_OBJECTIVES
    models = _GLOBAL_MODELS
    num_objectives = _GLOBAL_NUM_OBJECTIVES

    results = []
    for params in batch_params:
        params_array = np.array([params])
        predictions = []
        uncertainties = []
        for model in models:
            # For RandomForest: per-tree predictions give a rough uncertainty
            tree_preds = np.array([est.predict(params_array) for est in model.estimators_])
            predictions.append(float(np.mean(tree_preds, axis=0)[0]))
            uncertainties.append(float(np.std(tree_preds, axis=0)[0]))

        # Enforce vector lengths
        if len(predictions) < num_objectives:
            predictions += [0.0] * (num_objectives - len(predictions))
        else:
            predictions = predictions[:num_objectives]

        if len(uncertainties) < num_objectives:
            uncertainties += [0.0] * (num_objectives - len(uncertainties))
        else:
            uncertainties = uncertainties[:num_objectives]

        results.append({
            "params": params,
            "predictions": predictions,
            "uncertainties": uncertainties
        })
    return results

# ========================= Utilities & ML =========================
def create_fitness_class(num_objectives, weights):
    if 'FitnessMulti' in creator.__dict__:
        del creator.FitnessMulti
    creator.create("FitnessMulti", base.Fitness, weights=tuple(weights))
    return creator.FitnessMulti

def create_individual_class(num_objectives, weights):
    if 'Individual' in creator.__dict__:
        del creator.Individual
    creator.create("Individual", list, fitness=create_fitness_class(num_objectives, weights), uncertainty=list)

def load_data(num_objectives):
    uploaded_file = st.file_uploader("Upload Training Data Excel file", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            if len(df.columns) < num_objectives:
                st.error(f"Number of objectives ({num_objectives}) exceeds number of columns ({len(df.columns)}).")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None

@st.cache_data
def preprocess_and_train_with_cv(_df, num_objectives, n_estimators=50, cv_folds=5):
    X = _df.drop(_df.columns[-num_objectives:], axis=1)
    y_cols = _df.columns[-num_objectives:]
    y_list = [_df[col] for col in y_cols]

    cv = KFold(n_splits=max(2, min(cv_folds, len(X))), shuffle=True, random_state=42)

    models, cv_scores, importances = [], [], []
    for i, y in enumerate(y_list):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
        cv_scores.append(scores)
        model.fit(X, y)
        models.append(model)
        importances.append(model.feature_importances_)

        st.write(f"Objective {i+1} ({y_cols[i]}): R² {np.mean(scores):.2f} ± {np.std(scores):.2f} — {', '.join(f'{s:.2f}' for s in scores)}")

    if cv_scores:
        fig, ax = plt.subplots(figsize=(10,6), frameon=False)
        ax.boxplot(cv_scores)
        ax.set_xticklabels([str(c) for c in y_cols], rotation=45, ha="right")
        ax.set_ylabel("R²"); ax.set_title("Cross-Validation Performance"); ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout(); st.pyplot(fig)

    if importances:
        n = len(importances)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5), frameon=False)
        if n == 1: axes = [axes]
        for imp, ax, name in zip(importances, axes, y_cols):
            order = np.argsort(imp)
            ax.barh(np.array(X.columns)[order], np.array(imp)[order], color=cm.RdYlGn(np.linspace(0.2, 0.8, len(order))))
            ax.set_title(f"Sensitivity — {name}")
            ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout(); st.pyplot(fig)

    st.session_state["models"] = models
    return models, X.columns.tolist(), y_cols.tolist()

def parameter_combinations_generator(param_ranges):
    values = []
    for (mn, mx, step) in param_ranges.values():
        if mn == mx:
            values.append([mn])
        else:
            values.append(np.arange(mn, mx + step/2, step))
    total = 1
    for arr in values:
        total *= len(arr)
    return itertools.product(*values), total

def _batches(gen, size):
    batch = []
    for x in gen:
        batch.append(x)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch

def process_completed_results_into(hof, batch_results, tmp_csv, param_labels, objective_names):
    with open(tmp_csv, "a") as f:
        for r in batch_results:
            ind = creator.Individual(r["params"])
            ind.fitness.values = tuple(r["predictions"])
            ind.uncertainty = tuple(r["uncertainties"])
            ind.weighted_sum = 0.0
            hof.update([ind])
            row = [str(float(p)) for p in r["params"]]
            row += [f"{float(v):.6f}" for v in r["predictions"]]
            row += [f"{float(u):.6f}" for u in r["uncertainties"]]
            f.write(",".join(row) + "\n")

def create_dimension(label, values, mn, mx):
    if mn == mx:
        return dict(range=[mn-0.1, mx+0.1], label=label, values=values)
    return dict(range=[mn, mx], label=label, values=values)

# ========================= Optimization Engine =========================
def optimize_parameters_parallel(param_ranges, num_objectives, models, weights, output_path, objective_names,
                                 prefer_backend="auto"):
    """
    Robust optimization runner that prefers processes (spawn), falls back to threads or sequential
    if the deployed environment cannot pickle functions/models.
    """
    st.write("Starting optimization...")
    progress = st.progress(0.0); status = st.empty()
    start = time.time()

    hof = tools.ParetoFront()
    gen, total = parameter_combinations_generator(param_ranges)
    st.write(f"Total combinations: {total}")

    cores = st.session_state.get("num_cores", 2)
    # adaptive batch
    if total < 1000: bs = max(1, total // max(1, cores*2))
    elif total < 10000: bs = max(10, total // max(1, cores*4))
    elif total < 100000: bs = max(50, min(500, total // max(1, cores*8)))
    else: bs = max(100, min(500, total // max(1, cores*16)))
    bs = max(1, bs)

    params_labels = list(param_ranges.keys())
    fit_labels = list(objective_names)
    unc_labels = [f"Uncertainty_{x}" for x in objective_names]
    header = params_labels + fit_labels + unc_labels
    tmp_csv = "temp_results.csv"
    with open(tmp_csv, "w") as f: f.write(",".join(header) + "\n")

    processed = 0

    # Decide backend
    backend_order = []
    if prefer_backend in ("auto", "process"):
        backend_order.append("process")
    if prefer_backend in ("auto", "thread"):
        backend_order.append("thread")
    backend_order.append("sequential")

    last_error = None
    for backend in backend_order:
        try:
            _init_worker(models, num_objectives)  # set globals for ALL backends
            st.write(f"Backend: {backend} | batch={bs} | cores={cores}")
            if backend == "process":
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=cores, initializer=_init_worker, initargs=(models, num_objectives)) as pool:
                    for result_batch in pool.imap_unordered(_evaluate_batch_worker, _batches(gen, bs), chunksize=1):
                        process_completed_results_into(hof, result_batch, tmp_csv, params_labels, objective_names)
                        processed += len(result_batch)
                        prog = min(1.0, processed/total)
                        progress.progress(prog)
                        elapsed = time.time()-start
                        eta = (elapsed/prog - elapsed) if prog>0 else 0
                        status.text(f"{processed}/{total} | {elapsed:.1f}s | ETA {eta:.1f}s")
            elif backend == "thread":
                # Threads share memory; just use a ThreadPool
                with ThreadPool(processes=max(1, cores)) as pool:
                    for result_batch in pool.imap_unordered(_evaluate_batch_worker, _batches(gen, bs), chunksize=1):
                        process_completed_results_into(hof, result_batch, tmp_csv, params_labels, objective_names)
                        processed += len(result_batch)
                        prog = min(1.0, processed/total)
                        progress.progress(prog)
                        elapsed = time.time()-start
                        eta = (elapsed/prog - elapsed) if prog>0 else 0
                        status.text(f"{processed}/{total} | {elapsed:.1f}s | ETA {eta:.1f}s")
            else:
                # sequential
                for batch in _batches(gen, bs):
                    result_batch = _evaluate_batch_worker(batch)
                    process_completed_results_into(hof, result_batch, tmp_csv, params_labels, objective_names)
                    processed += len(batch)
                    prog = min(1.0, processed/total)
                    progress.progress(prog)
                    elapsed = time.time()-start
                    eta = (elapsed/prog - elapsed) if prog>0 else 0
                    status.text(f"{processed}/{total} | {elapsed:.1f}s | ETA {eta:.1f}s")
            last_error = None
            break
        except (PicklingError, AttributeError, RuntimeError, OSError) as e:
            last_error = e
            st.warning(f"{backend.title()} backend failed: {e}")
            continue
        except Exception as e:
            last_error = e
            st.warning(f"{backend.title()} backend error: {e}")
            continue

    if last_error and processed == 0:
        st.error(f"All backends failed: {last_error}")
        return None

    progress.progress(1.0)

    # finalize CSV -> DataFrame and compute weighted sum
    df = pd.read_csv(tmp_csv)
    if "Weighted Sum" not in df.columns:
        ws = np.zeros(len(df))
        for obj, w in zip(fit_labels, weights):
            col = df[obj].astype(float)
            vmin, vmax = col.min(), col.max()
            if vmax == vmin:
                norm = np.zeros_like(col)
            elif w < 0:
                norm = w * (vmax - col) / (vmax - vmin)
            else:
                norm = w * (col - vmin) / (vmax - vmin)
            ws += norm
        df["Weighted Sum"] = ws
        # also push to HOF objects for UI ordering
        for ind, wsum in zip(hof, ws):
            ind.weighted_sum = float(wsum)

    # save
    try:
        if len(df) <= 1_048_576:
            df.to_excel(output_path, index=False)
            st.success(f"Results saved to {output_path}")
            st.session_state["output_path"] = output_path
        else:
            csv_path = output_path.replace(".xlsx", ".csv")
            df.to_csv(csv_path, index=False)
            st.warning(f"Too many rows for Excel; saved CSV to {csv_path}")
            st.session_state["output_path"] = csv_path
    except Exception as e:
        st.error(f"Excel save failed: {e}. Saving CSV instead.")
        csv_path = output_path.replace(".xlsx", ".csv")
        df.to_csv(csv_path, index=False)
        st.success(f"Results saved to {csv_path}")
        st.session_state["output_path"] = csv_path

    try: os.remove(tmp_csv)
    except: pass

    st.write(f"Done in {time.time()-start:.2f}s. Evaluated {processed}/{total}. Pareto size = {len(hof)}")
    return hof

# ========================= Visualizations =========================
def visualize_pareto_front(hof, objective_names, param_names, weights):
    try:
        if not hof:
            st.info("No solutions to display.")
            return
        data = []
        for i, ind in enumerate(hof):
            row = {}
            for j, p in enumerate(param_names):
                row[p] = float(ind[j])
            for j, o in enumerate(objective_names):
                row[o] = float(ind.fitness.values[j])
            row["Weighted Sum"] = float(getattr(ind, "weighted_sum", 0.0))
            row["Solution ID"] = i
            data.append(row)
        df = pd.DataFrame(data)
        dims = []
        for p in param_names:
            dims.append(create_dimension(p, df[p], df[p].min(), df[p].max()))
        for o in objective_names:
            dims.append(create_dimension(o, df[o], df[o].min(), df[o].max()))
        dims.append(create_dimension("Weighted Sum", df["Weighted Sum"], df["Weighted Sum"].min(), df["Weighted Sum"].max()))
        with st.expander("Optimization Result (Pareto Front)", expanded=False):
            fig = go.Figure(data=go.Parcoords(
                line=dict(color=df["Weighted Sum"], colorscale="Tealrose", showscale=True, colorbar=dict(title="Weighted Sum", nticks=11)),
                dimensions=dims,
                unselected=dict(line=dict(opacity=0))
            ))
            fig.update_layout(height=600, margin=dict(l=50,r=50,t=60,b=40))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Pareto visualization error: {e}")

def visualize_selected_solution(solution, param_names, objective_names, weights, hof):
    try:
        st.subheader("Selected Solution Details")
        c1, c2 = st.columns(2)
        with c1:
            st.write("Parameters")
            st.table(pd.DataFrame({"Parameter": param_names, "Value": [f"{float(v):.3f}" for v in solution]}))
        with c2:
            st.write("Objectives")
            st.table(pd.DataFrame({
                "Objective": objective_names,
                "Value": [f"{float(v):.3f}" for v in solution.fitness.values],
                "(±)": [f"{float(u):.3f}" for u in getattr(solution, "uncertainty", [0]*len(objective_names))],
                "Weight": [f"{float(w):.2f}" for w in weights],
            }))
    except Exception as e:
        st.error(f"Selected solution view error: {e}")

def visualize_parameter_trends(hof, param_names, objective_names):
    with st.expander("Trend Analysis", expanded=False):
        try:
            if not hof:
                st.info("No data for trends.")
                return
            P = np.array([[float(v) for v in ind] for ind in hof])
            O = np.array([[float(v) for v in ind.fitness.values] for ind in hof])
            for p_idx, pname in enumerate(param_names):
                st.subheader(f"Parameter: {pname}")
                fig, axes = plt.subplots(1, len(objective_names), figsize=(15,4), frameon=False)
                axes = [axes] if len(objective_names)==1 else axes
                for o_idx, (oname, ax) in enumerate(zip(objective_names, axes)):
                    x, y = P[:,p_idx], O[:,o_idx]
                    ax.scatter(x,y,alpha=0.5)
                    if len(x) > 1:
                        try:
                            deg = 2 if len(x)>5 else 1
                            z = np.polyfit(x,y,deg); p = np.poly1d(z)
                            xx = np.linspace(x.min(), x.max(), 100); ax.plot(xx, p(xx), "r--")
                        except: pass
                    ax.set_xlabel(pname); ax.set_ylabel(oname); ax.grid(True, linestyle="--", alpha=0.7)
                plt.tight_layout(); st.pyplot(fig)
        except Exception as e:
            st.error(f"Trend analysis error: {e}")

def display_optimization_results():
    st.title("Optimization Results Explorer")
    hof = st.session_state.get("hof")
    if not hof:
        st.error("No optimization results found.")
        return
    param_names = st.session_state["param_names"]
    objective_names = st.session_state["objective_names"]
    weights = st.session_state["weights"]

    # ensure weighted_sum set
    for ind in hof:
        if not hasattr(ind, "weighted_sum"):
            ind.weighted_sum = float(np.sum(ind.fitness.values))

    visualize_pareto_front(hof, objective_names, param_names, weights)
    visualize_parameter_trends(hof, param_names, objective_names)

    sorted_idx = sorted(range(len(hof)), key=lambda i: hof[i].weighted_sum)
    with st.expander("Interactive Solution Explorer", expanded=False):
        sel = st.selectbox("Select solution:", options=list(range(len(sorted_idx))),
                           format_func=lambda i: f"Solution {i+1} (Score {hof[sorted_idx[i]].weighted_sum:.3f})")
        visualize_selected_solution(hof[sorted_idx[sel]], param_names, objective_names, weights, hof)

# ========================= Explorer =========================
def calculate_explorer_weighted_sum(obj_values, weights):
    try:
        n, m = obj_values.shape
        scores = np.zeros((n, m))
        for j, w in enumerate(weights):
            col = obj_values[:, j]
            vmin, vmax = np.min(col), np.max(col)
            if vmax == vmin:
                scores[:, j] = abs(w)
            elif w < 0:
                scores[:, j] = abs(w) * (vmax - col) / (vmax - vmin)
            else:
                scores[:, j] = abs(w) * (col - vmin) / (vmax - vmin)
        return np.sum(scores, axis=1)
    except Exception:
        return np.zeros(obj_values.shape[0])

def explore_uploaded_results():
    st.title("Explorer")
    st.write("Upload and explore previously computed optimization results")
    num_objectives = st.number_input("Enter Number of Objectives", min_value=1, value=4, key="expl_num_obj")
    uploaded = st.file_uploader("Upload Optimization Results (.xlsx or .csv)", type=["xlsx","csv"], key="expl_upload")
    if uploaded is None:
        st.info("Upload a results file to continue.")
        return
    try:
        df = pd.read_excel(uploaded, engine="openpyxl") if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    ws_col = "Weighted Sum" if "Weighted Sum" in df.columns else None
    unc_cols = [c for c in df.columns if c.startswith("Uncertainty_")]
    exclude = unc_cols + ([ws_col] if ws_col else [])
    core = [c for c in df.columns if c not in exclude]
    if len(core) < num_objectives:
        st.error(f"Objectives ({num_objectives}) exceed available columns ({len(core)}).")
        return

    param_names = core[:-num_objectives]
    objective_names = core[-num_objectives:]
    weights = [-1.0 for _ in objective_names]

    with st.expander("Setup Objectives Weights", expanded=False):
        cols = st.columns(3)
        for i, obj in enumerate(objective_names):
            with cols[i%3]:
                weights[i] = st.number_input(f"Weight for {obj}", value=-1.0, help="Negative=minimize, Positive=maximize", key=f"w_{obj}")

    with st.expander("Dataset Overview", expanded=False):
        st.write(f"**Total Solutions:** {len(df)}")
        st.write(f"**Parameters:** {', '.join(param_names)}")
        st.write(f"**Objectives:** {', '.join(objective_names)}")

    with st.expander("Estimate Objectives Results for Custom Parameters Values", expanded=False):
        cols = st.columns(3)
        inputs = {}
        for i, p in enumerate(param_names):
            with cols[i%3]:
                inputs[p] = st.number_input(p, value=float(df[p].median()) if p in df.columns else 0.0, key=f"custom_{p}")
        if st.button("Estimate Objectives Results", key="estimate_btn_expl"):
            try:
                X = df[param_names]; models = []
                for obj in objective_names:
                    m = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X, df[obj])
                    models.append(m)
                x_in = np.array([[inputs[p] for p in param_names]])
                preds = [m.predict(x_in)[0] for m in models]

                vals, labels, mins, maxs = [], [], [], []
                for obj, pred in zip(objective_names, preds):
                    arr = df[obj].values; mn, mx = float(np.min(arr)), float(np.max(arr))
                    norm = 50.0 if mx==mn else (float(pred)-mn)/(mx-mn)*100.0
                    vals.append(norm); labels.append(obj); mins.append(mn); maxs.append(mx)
                vals += vals[:1]; labels += labels[:1]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=vals, theta=labels, fill='toself', name='Predictions'))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])), showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)

                st.table(pd.DataFrame({"Objective": objective_names, "Estimated Value": [f"{p:.2f}" for p in preds], "Min": [f"{m:.2f}" for m in mins], "Max": [f"{m:.2f}" for m in maxs]}))
            except Exception as e:
                st.error(f"Estimation failed: {e}")

    with st.expander("All Computed Solutions (Interactive)", expanded=False):
        df_disp = df.copy()
        if "Weighted Sum" not in df_disp.columns:
            obj_matrix = np.column_stack([df_disp[o].astype(float).values for o in objective_names])
            df_disp["Weighted Sum"] = calculate_explorer_weighted_sum(obj_matrix, weights)

        filters = {}
        st.write("### Filter by Parameter Values")
        for p in param_names:
            if p in df_disp.columns:
                mn, mx = float(df_disp[p].min()), float(df_disp[p].max())
                filters[p] = st.slider(p, mn, mx, (mn, mx), key=f"f_{p}")
        st.write("### Filter by Objective Values")
        for o in objective_names:
            if o in df_disp.columns:
                mn, mx = float(df_disp[o].min()), float(df_disp[o].max())
                filters[o] = st.slider(o, mn, mx, (mn, mx), key=f"f_{o}")
        mn, mx = float(df_disp["Weighted Sum"].min()), float(df_disp["Weighted Sum"].max())
        filters["Weighted Sum"] = st.slider("Weighted Sum", mn, mx, (mn, mx), key="f_ws")

        fdf = df_disp.copy()
        for col, (low, high) in filters.items():
            fdf = fdf[(fdf[col] >= low) & (fdf[col] <= high)]
        st.write(f"Filtered to {len(fdf)} solutions.")

        dims = []
        for col in param_names + objective_names:
            dims.append(create_dimension(col, fdf[col], fdf[col].min(), fdf[col].max()))
        dims.append(create_dimension("Weighted Sum", fdf["Weighted Sum"], fdf["Weighted Sum"].min(), fdf["Weighted Sum"].max()))
        fig = go.Figure(data=go.Parcoords(
            line=dict(color=fdf["Weighted Sum"], colorscale="Tealrose", showscale=True),
            dimensions=dims, unselected=dict(line=dict(opacity=0))
        ))
        fig.update_layout(height=600, margin=dict(l=50,r=50,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)

        if len(fdf) > 0:
            export_path = st.text_input("Export filtered solutions to:", value="filtered_solutions.xlsx")
            if st.button("Export Filtered Solutions", use_container_width=True):
                try:
                    if export_path.endswith(".xlsx"):
                        fdf.to_excel(export_path, index=False)
                    else:
                        fdf.to_csv(export_path, index=False)
                    st.success(f"Exported to {export_path}")
                except Exception as e:
                    st.error(f"Export failed: {e}")

# ========================= App State & Main =========================
def initialize_session_state():
    defaults = {
        "models": [], "optimization_complete": False, "hof": None,
        "param_names": [], "objective_names": [], "weights": [],
        "selected_solution_index": 0, "active_tab": "Training",
        "num_cores": 2, "output_path": "evaluated_solutions.xlsx"
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def main():
    initialize_session_state()

    with st.sidebar.expander("Setup", expanded=False):
        st.header("Setup")
        n_estimators = st.slider("Number of Trees (RF)", 10, 200, 50)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        st.session_state["num_cores"] = st.slider("CPU Cores", 1, 8, 2)
        backend = st.selectbox("Optimization backend", ["auto", "process", "thread", "sequential"], index=0,
                               help="Use 'thread' or 'sequential' if your deployment has pickling issues.")

    tab_train, tab_results, tab_explorer = st.tabs(["Training", "Results", "Explorer"])

    with tab_results:
        if st.session_state["optimization_complete"] and st.session_state["hof"]:
            display_optimization_results()
        else:
            st.info("Run optimization in the Training tab to see results here.")
            if st.button("Go to Training Tab"):
                st.session_state["active_tab"] = "Training"
                st.rerun()

    with tab_explorer:
        explore_uploaded_results()

    with tab_train:
        st.header("Training and Optimization")
        num_objectives = st.number_input("Enter Number of Objectives", min_value=1, value=4)
        df = load_data(num_objectives)
        if df is None: return

        if len(df.columns) > 0 and num_objectives > len(df.columns):
            st.warning(f"Adjusted number of objectives to {len(df.columns)} to match file.")
            num_objectives = len(df.columns)

        with st.expander("Model Training and Validation", expanded=False):
            X = df.drop(df.columns[-num_objectives:], axis=1)
            y = [df[col] for col in df.columns[-num_objectives:]]
            objective_names = df.columns[-num_objectives:].tolist()

            models, param_names, objective_names = preprocess_and_train_with_cv(df, num_objectives, n_estimators, cv_folds)

            if models and y:
                st.subheader("Learning Curves")
                try:
                    fig, axes = plt.subplots(1, len(y), figsize=(15, 5), frameon=False)
                    if len(y)==1: axes = [axes]
                    folds = max(2, min(cv_folds, len(X)))
                    for yi, model, ax, oname in zip(y, models, axes, objective_names):
                        try:
                            train_sizes, train_scores, test_scores = learning_curve(
                                model, X, yi, cv=folds, n_jobs=-1,
                                train_sizes=np.linspace(0.1, 1.0, min(10, len(X))),
                                scoring='r2'
                            )
                            ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Train')
                            ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label='CV')
                            ax.set_title(oname); ax.set_xlabel('Samples'); ax.set_ylabel('R²'); ax.legend(); ax.grid(True, linestyle='--', alpha=0.7)
                        except Exception as e:
                            ax.text(0.5, 0.5, f"LC error: {e}", ha='center', va='center')
                    plt.tight_layout(); st.pyplot(fig)
                except Exception as e:
                    st.error(f"Learning curve error: {e}")

        with st.expander("Setup Training and run Optimization", expanded=False):
            st.subheader("Define Parameters Ranges")
            param_ranges = {}
            cols = st.columns(3)
            for i, p in enumerate(param_names):
                with cols[i%3]:
                    mn = st.number_input(f"{p} - Min", value=float(df[p].min()), key=f"min_{p}")
                    mx = st.number_input(f"{p} - Max", value=float(df[p].max()), key=f"max_{p}")
                    step = st.number_input(f"{p} - Step", value=max(0.1, (float(df[p].max())-float(df[p].min()))/10), key=f"step_{p}")
                    param_ranges[p] = (mn, mx, step)

            st.subheader("Define Objective Weights")
            weights = []
            cols = st.columns(3)
            for i, obj in enumerate(objective_names):
                with cols[i%3]:
                    weights.append(st.number_input(f"{obj} weight", value=-1.0, help="Negative=minimize, Positive=maximize", key=f"w_{i}"))

            output_path = st.text_input("Output file path (.xlsx)", value="evaluated_solutions.xlsx")

            if st.button("Optimize Parameters"):
                create_individual_class(num_objectives, weights)
                st.session_state["weights"] = weights

                models = st.session_state.get("models", [])
                if not models:
                    st.error("No trained models found. Train models first.")
                    return

                with st.spinner("Optimizing parameters..."):
                    hof = optimize_parameters_parallel(
                        param_ranges, num_objectives, models, weights, output_path, objective_names,
                        prefer_backend=backend
                    )

                if hof:
                    st.session_state["hof"] = hof
                    st.session_state["param_names"] = list(param_names)
                    st.session_state["objective_names"] = list(objective_names)
                    st.session_state["optimization_complete"] = True
                    st.session_state["active_tab"] = "Results"
                    st.rerun()
                else:
                    st.warning("Optimization returned no results.")

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
