import streamlit as st
import pandas as pd
import numpy as np
from PIL.ImageChops import lighter
from deap import base, creator, tools
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, learning_curve
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import pickle
import tempfile
from functools import partial
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.cm as cm

st.set_page_config(
    initial_sidebar_state="collapsed",  # Optional
    page_title="WSGT_PAMO Machine Learning",
    page_icon="Pamo_Icon_White.png",
    layout="wide"
)

st.sidebar.image("Pamo_Icon_Black.png", width=80)
st.sidebar.write("## WSGT_PAMO")
st.sidebar.write("Version 1.0.0")
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


# Function to preprocess data and train model with cross-validation
@st.cache_data
def preprocess_and_train_with_cv(_df, num_objectives, n_estimators=50, cv_folds=5):
    try:
        X = _df.drop(_df.columns[-num_objectives:], axis=1)
        y = [_df[col] for col in _df.columns[-num_objectives:]]

        # Define cross-validation strategy
        cv = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)

        models = []
        cv_scores = []
        feature_importances = []

        # For each objective
        for i, y_i in enumerate(y):
            # Initialize model
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

            # Perform cross-validation
            scores = cross_val_score(model, X, y_i, cv=cv, scoring='r2')
            cv_scores.append(scores)

            # Train final model on all data
            model.fit(X, y_i)
            models.append(model)

            # Store feature importances
            feature_importances.append(model.feature_importances_)

            # Print cross-validation results
            st.write(
                f"Objective {i + 1} ({_df.columns[-num_objectives + i]}) - Cross-validation R² scores: {[f'{s:.2f}' for s in scores]}")
            st.write(f"Mean R²: {np.mean(scores):.2f} (±{np.std(scores):.2f})")

        # Visualize cross-validation results
        if cv_scores:
            fig, ax = plt.subplots(figsize=(10, 6), frameon=False)
            ax.boxplot(cv_scores)
            ax.set_xticklabels([f"{_df.columns[-num_objectives + i]}" for i in range(len(y))], rotation=45, ha='right')
            ax.set_ylabel("R² Score")
            ax.set_title("Cross-Validation Performance by Objective")
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

        # Visualize feature sensitivity
        if feature_importances:
            fig, axes = plt.subplots(1, len(y), figsize=(15, 5), sharey=False, frameon=False)
            if len(y) == 1:
                axes = [axes]

            for i, (importances, ax) in enumerate(zip(feature_importances, axes)):
                # Sort indices by importance
                sorted_indices = np.argsort(importances)
                sorted_importances = importances[sorted_indices]
                sorted_names = X.columns[sorted_indices]

                # Create horizontal bars with color gradient
                colors = cm.RdYlGn_r(sorted_importances / max(sorted_importances) if max(sorted_importances) > 0 else 0)

                # Plot horizontal bars
                y_pos = np.arange(len(sorted_indices))
                ax.barh(y_pos, sorted_importances, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(sorted_names)
                ax.set_title(f"Sensitivity - {_df.columns[-num_objectives + i]}")
                ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig)

        # Store models directly in session state
        st.session_state['models'] = models

        # Test prediction with a sample
        if X.shape[0] > 0:
            sample = X.iloc[0].values
            st.write("### Testing Model Predictions")
            st.write(f"Sample input: {sample}")

            for i, model in enumerate(models):
                pred = model.predict([sample])[0]
                st.write(f"Model {i + 1} prediction: {pred:.2f}")

        return models, X.columns, _df.columns[-num_objectives:]

    except Exception as e:
        st.error(f"Error in model training: {e}")
        import traceback
        st.code(traceback.format_exc())
        return [], [], []


# Visualize model performance with learning curves
@st.cache_data
def visualize_model_performance(X, y, _models, objective_names, cv_folds=5):
    try:
        # Ensure we have models to visualize
        if not _models or not y:
            st.warning("No models or data available for learning curves.")
            return

        # Create a figure for learning curves (without frame)
        fig, axes = plt.subplots(1, len(y), figsize=(15, 5), frameon=False)
        if len(y) == 1:
            axes = [axes]

        # Adjust cv_folds if necessary
        cv_folds = min(cv_folds, len(X))

        # For each objective
        for i, (y_i, model, ax, obj_name) in enumerate(zip(y, _models, axes, objective_names)):
            try:
                # Split data for plotting learning curve
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X, y_i, cv=cv_folds, n_jobs=-1,
                    train_sizes=np.linspace(0.1, 1.0, min(10, len(X))),
                    scoring='r2'
                )

                # Calculate mean and std
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)

                # Plot learning curve
                ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
                ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
                ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
                ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
                ax.set_title(f'Learning Curve - {obj_name}')
                ax.set_xlabel('Training Examples')
                ax.set_ylabel('R² Score')
                ax.legend(loc='best')
                ax.grid(True, linestyle='--', alpha=0.7)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                ax.set_title(f'Learning Curve - {obj_name} (Error)')

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error visualizing learning curves: {e}")


# Generator function for parameter combinations - lazy evaluation
def parameter_combinations_generator(param_ranges):
    """Generate parameter combinations on-demand without storing all in memory"""
    # Extract parameter values
    param_values = []
    for min_val, max_val, step in param_ranges.values():
        # Handle case where min == max
        if min_val == max_val:
            param_values.append([min_val])
        else:
            param_values.append(
                np.arange(min_val, max_val + step / 2, step))  # Add step/2 to avoid floating point issues

    # Calculate total combinations without generating them
    total_combinations = 1
    for values in param_values:
        total_combinations *= len(values)

    # Return generator and total count
    return itertools.product(*param_values), total_combinations


# Optimized function to evaluate a batch of parameter combinations
def evaluate_batch(batch_params, models, num_objectives):
    results = []

    for params in batch_params:
        # Convert params to numpy array for prediction
        params_array = np.array([params])

        # Get predictions and uncertainties for all models
        predictions = []
        uncertainties = []

        for model in models:
            # For Random Forest, estimate uncertainty using predictions from individual trees
            tree_preds = np.array([tree.predict(params_array) for tree in model.estimators_])

            # Mean prediction across trees
            mean_pred = np.mean(tree_preds, axis=0)[0]

            # Standard deviation as uncertainty measure
            std_pred = np.std(tree_preds, axis=0)[0]

            predictions.append(float(mean_pred))
            uncertainties.append(float(std_pred))

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

        # Format for display
        rounded_fitness = [f"{float(fit_val):.2f}" for fit_val in predictions]
        rounded_uncertainty = [f"{float(unc_val):.2f}" for unc_val in uncertainties]

        # Add to results
        results.append({
            'params': params,
            'predictions': predictions,
            'uncertainties': uncertainties,
            'rounded_fitness': rounded_fitness,
            'rounded_uncertainty': rounded_uncertainty
        })

    return results


# Helper function to process completed async results
def process_completed_results(async_results, hof, temp_results_file,
                              processed_count, total_combinations,
                              progress_bar, status_text, start_time):
    """Process any completed async results"""
    # Check which results are ready
    completed_indices = []
    for i, (async_result, batch_size) in enumerate(async_results):
        if async_result.ready():
            completed_indices.append(i)

            # Get the result
            result_batch = async_result.get()

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
def optimize_parameters_parallel(param_ranges, num_objectives, _models, weights, output_path, objective_names):
    try:
        st.write("Starting optimization process with streaming parallel processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        hof = tools.ParetoFront()

        # Get parameter combinations generator and total count
        param_generator, total_combinations = parameter_combinations_generator(param_ranges)

        st.write(f"Total parameter combinations to evaluate: {total_combinations}")

        # Use the number of cores from session state
        num_cores = st.session_state.get('num_cores', 8)
        st.write(f"Using {num_cores} CPU cores for parallel processing")

        # Calculate optimal batch size based on total combinations
        if total_combinations < 1000:
            batch_size = max(1, total_combinations // (num_cores * 2))
        elif total_combinations < 10000:
            batch_size = max(10, total_combinations // (num_cores * 4))
        elif total_combinations < 100000:
            batch_size = max(50, min(500, total_combinations // (num_cores * 8)))
        else:
            batch_size = max(100, min(500, total_combinations // (num_cores * 16)))

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

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_cores) as pool:
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
                    async_result = pool.apply_async(evaluate_batch, args=(current_batch, _models, num_objectives))
                    async_results.append((async_result, len(current_batch)))

                    # Reset current batch
                    current_batch = []

                # Process any completed results
                processed_count = process_completed_results(
                    async_results, hof, temp_results_file,
                    processed_count, total_combinations,
                    progress_bar, status_text, start_time
                )

            # Submit any remaining combinations
            if current_batch:
                batch_count += 1
                batches.append(current_batch)

                # Submit batch for async processing
                async_result = pool.apply_async(evaluate_batch, args=(current_batch, _models, num_objectives))
                async_results.append((async_result, len(current_batch)))

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
                    all_normalized_values.append(normalized_pred)

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
                    "Normalized (%)": [f"{n:.1f}%" for n in all_normalized_values]
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
                    export_path = st.text_input("Export filtered solutions to:", value="filtered_solutions.xlsx")

                    if st.button("Export Filtered Solutions", use_container_width=True):
                        try:
                            if export_path.endswith(".xlsx"):
                                filtered_df.to_excel(export_path, index=False)
                            else:
                                filtered_df.to_csv(export_path, index=False)
                            st.success(f"Filtered solutions exported to {export_path}")
                        except Exception as e:
                            st.error(f"Error exporting filtered solutions: {e}")

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
    with st.expander("Setup Objectives Weights",expanded=False):
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
    with st.expander("Dataset Overview",expanded=False):
        st.write("### Dataset Overview")
        st.write(f"**Total Solutions:** {len(df)}")
        st.write(f"**Parameters:** {', '.join(param_names)}")
        st.write(f"**Objectives:** {', '.join(objective_names)}")

    # === Estimate Objectives for Custom Parameter Combination in Explorer ===
    with st.expander("Estimate Objectives Results for Custom Parameters Values", expanded=False):
        st.write("## Estimate Objectives Results for Custom Parameters Values")
        st.write(
            "Here you can estimate what is expected to be the result for each individual objective for entered parameter values")

        st.write("### Enter parameter values to simulate:")

        input_values = {}
        col1, col2, col3 = st.columns(3)
        for i, param in enumerate(param_names):
            col = [col1, col2, col3][i % 3]
            with col:
                input_values[param] = st.number_input(f"{param}", key=f"explorer_custom_input_{i}")

        input_array = np.array([[input_values[p] for p in param_names]])

        if st.button("Estimate Objectives Results", key="explorer_estimate_btn"):
            # Train models with uploaded data
            try:
                # Prepare training data from uploaded file
                X_train = df[param_names]
                y_train = [df[obj] for obj in objective_names]

                st.write("Training models with uploaded data...")
                models = []

                # Train a model for each objective
                for i, y_i in enumerate(y_train):
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_i)
                    models.append(model)

                st.subheader("Estimated Objective Values")
                preds = [model.predict(input_array)[0] for model in models]
                df_estimates = pd.DataFrame({
                    "Objective": objective_names,
                    "Estimated Value": [f"{p:.2f}" for p in preds]
                })



                # RADAR CHART IMPLEMENTATION - REPLACES GAUGE CHARTS
                # Collect all data for radar chart
                all_pred_values = []
                all_obj_names = []
                all_obj_min = []
                all_obj_max = []
                all_normalized_values = []

                # Collect values from the loop and normalize each objective individually
                for i, (obj, pred) in enumerate(zip(objective_names, preds)):
                    obj_values = df[obj].values
                    obj_min, obj_max = obj_values.min(), obj_values.max()

                    # Normalize the prediction value to 0-100 scale for this specific objective
                    if obj_max > obj_min:
                        normalized_pred = ((float(pred) - obj_min) / (obj_max - obj_min)) * 100
                    else:
                        normalized_pred = 50  # If min == max, put it in the middle

                    all_pred_values.append(float(pred))  # Keep original values for hover/display
                    all_obj_names.append(str(obj))
                    all_obj_min.append(obj_min)
                    all_obj_max.append(obj_max)
                    all_normalized_values.append(normalized_pred)

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
                    "Normalized (%)": [f"{n:.1f}%" for n in all_normalized_values]
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
                export_path = st.text_input("Export filtered solutions to:",
                                            value="uploaded_filtered_solutions.xlsx",
                                            key="explorer_export_path")

                if st.button("Export Filtered Solutions", key="explorer_export_btn", use_container_width=True):
                    try:
                        if export_path.endswith(".xlsx"):
                            filtered_df.to_excel(export_path, index=False)
                        else:
                            filtered_df.to_csv(export_path, index=False)
                        st.success(f"Filtered solutions exported to {export_path}")
                    except Exception as e:
                        st.error(f"Error exporting filtered solutions: {e}")

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
            n_estimators = st.slider("Number of Trees in Random Forest", min_value=10, max_value=200, value=50,help="Only change if you are sure what you are doing")
            cv_folds = st.slider("Cross-Validation Folds", min_value=3, max_value=10, value=5,help="Only change if you are sure what you are doing")
            st.session_state['num_cores'] = st.slider("Number of CPU Cores", min_value=1, max_value=16, value=8, help="Only change if you are sure what you are doing")

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

                models, param_names, objective_names = preprocess_and_train_with_cv(df, num_objectives, n_estimators,
                                                                                    cv_folds)

                if models and y:
                    st.subheader("Learning Curves")
                    objective_names_list = list(objective_names)
                    visualize_model_performance(X, y, models, objective_names_list, cv_folds)

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
                        hof = optimize_parameters_parallel(param_ranges, num_objectives, models, weights, output_path,
                                                           objective_names_list)

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