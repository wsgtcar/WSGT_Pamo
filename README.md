# PAMO 1.1.10 - README
Machine Learning Based Design Investigation and Optimization Tool

## Table of contents
1. What PAMO is  
2. Core concept: surrogate modeling  
3. Key features  
4. Installation and setup  
5. Data requirements (training dataset)  
6. Model Strategy Selection (Auto vs Manual; Ridge vs RF vs Hybrid)  
7. Precision, reliability, and overfitting  
8. Setup controls  
9. Learning curves  
10. Caching behavior (cached version)  
11. Troubleshooting  
12. Best practices for consulting use  
13. Governance and data strategy  
14. Model strategy quick cheat sheet  

---

## 1. What PAMO is

**PAMO** is a Streamlit-based optimization toolkit that learns **surrogate (proxy) models** from simulation or measured building-performance data and then uses those models to:

- **Predict multiple objectives** (e.g., energy, comfort, daylight, cost) from a set of design/operation parameters.
- **Optimize parameters** to find **Pareto-optimal** solutions in a **multi-objective** context.
- Provide an **Explorer** workflow to evaluate "what-if" parameter sets quickly without rerunning expensive simulations.

PAMO is intended as a **decision-support accelerator** for building-performance consultants - not a replacement for validated simulation workflows.

---

## 2. Core concept: surrogate modeling for building performance

In building performance work, many tasks require repeated evaluation of outcomes:

- Heating/cooling loads and energy use
- Thermal comfort metrics (e.g., overheating hours)
- Daylight metrics (e.g., SDA/UDI)
- Cost / carbon / KPI composites

A full simulation (IESVE, EnergyPlus, IDA ICE, etc.) can take minutes to hours per variant.

PAMO trains data-driven surrogate models on an uploaded dataset of prior runs:

- **Inputs**: your parameters (design variables or operational settings)
- **Outputs**: your objectives (energy, comfort, cost, etc.)

Once trained, surrogate models can predict objective values in milliseconds, enabling:

- Rapid sensitivity testing
- Optimization over large candidate spaces
- Interactive exploration

---

## 3. Key features

### Training tab
- Upload training data.
- Define parameter names and ranges.
- Define objective names and (optionally) weights.
- Train surrogate models using the **Model Strategy Selection** engine (Section 6).
- Validate with cross-validation metrics and learning curves.
- Run optimization to generate Pareto solutions.

### Explorer tab
- Predict objectives for custom parameter values.
- Display radar chart, ranges, normalized (%) and percentiles.
- Optionally, train Explorer-only surrogates on an uploaded Explorer dataset (depending on workflow).

### Results
- Pareto front extraction
- Solution ranking
- Exportable results

---

## 4. Installation and setup

### Recommended environment
- Python 3.10+ (3.11 typically works)
- Windows supported (learning curve routine avoids Windows multiprocessing issues)

### Install dependencies
From a terminal inside your virtual environment:

```bash
pip install streamlit pandas numpy scikit-learn scipy matplotlib plotly openpyxl joblib
```

### Run PAMO
From the folder containing the script:

```bash
streamlit run PAMO_1.1.10.py
```

---

## 5. Data requirements (training dataset)

### 5.1 Structure
Your training dataset must be a table with:

- **Parameter columns** (inputs): e.g., `U_wall`, `WWR`, `ACH`, `setpoint`, etc.
- **Objective columns** (outputs): e.g., `Heizlast`, `Uebergradstunden`, `SDA`, `Kosten`

Each row is one evaluated case (one simulation run or measured operational snapshot).

### 5.2 Data quality expectations (critical)
Model accuracy and optimization reliability depend heavily on data quality.

**Strongly recommended**
- Consistent simulation settings across all cases (weather file, schedules, internal gains, HVAC topology, controls).
- Consistent naming/units.
- Sufficient coverage of the parameter space (avoid training only near one "design point").
- Remove failed simulation runs (outliers due to convergence issues, wrong boundary conditions).

**Common issues that reduce precision**
- Mixed modeling assumptions inside one dataset (e.g., different ventilation schedules across runs without encoding them as parameters).
- Hidden categorical changes (system type changed but not included as an input feature).
- Too few samples for too many parameters.

---

## 6. Model Strategy Selection (Auto vs Manual; per objective)

PAMO is a **multi-objective** system: each objective can behave differently. Example:

- Heating load might be relatively smooth and near-linear with respect to envelope parameters.
- Overheating hours might show thresholds, regime changes, and strong nonlinear effects (controls, solar gains, thermal mass).

Therefore, PAMO selects the best modeling strategy **per objective independently**.

### 6.1 Available model strategies

#### A) Poly(deg=2) + Ridge (Polynomial Ridge)
**What it is**
- A pipeline that expands inputs into degree-2 polynomial features (linear terms, squares, pairwise interactions) and fits **Ridge regression**.

**Strengths**
- Stable on smaller datasets
- Captures smooth trends and low-order interactions
- Regularization reduces overfitting risk
- Often more plausible behavior when slightly extrapolating

**Typical building-performance use**
- Energy KPIs in stable regimes
- Costs that are approximately additive/interaction-based
- Metrics that vary smoothly with parameters

#### B) Random Forest Regression (RF)
**What it is**
- An ensemble of decision trees; prediction is the average across many trees.

**Strengths**
- Captures nonlinear and threshold behavior well
- Learns complex interactions automatically
- Often strong when dataset coverage is good

**Risks**
- Can overfit if dataset is small or sparse
- Usually poor at extrapolation outside the training space

**Typical building-performance use**
- Comfort metrics with controls and regime changes
- Discontinuous responses (economizer enable/disable, setpoint deadbands)
- Metrics with complex nonlinear dependencies

#### C) Hybrid: Poly+Ridge baseline + RF residual corrector
**What it is**
- First fit Poly+Ridge to capture the global trend.
- Compute residuals: `residual = y - y_ridge`.
- Train an RF model on residuals to capture nonlinearity not captured by Ridge.
- Final prediction: `y_hat = y_ridge_hat + residual_rf_hat`.

**Strengths**
- Combines Ridge stability with RF flexibility
- Often best when objectives have both a smooth trend and localized nonlinear effects

**Typical building-performance use**
- Mixed-regime KPIs (physics + control logic)
- Cases where RF-only feels unstable but Ridge-only is not accurate enough

---

### 6.2 Auto vs Manual

#### Auto (recommended for most users)
PAMO automatically:
1. Trains/evaluates **Ridge**, **RF**, and **Hybrid** for each objective.
2. Uses **K-Fold cross-validation** for out-of-sample estimates.
3. Compares models using **NMAE** (Normalized MAE) per objective:

`NMAE = MAE / (max(y) - min(y) + epsilon)`

4. Selects the best strategy per objective with guardrails:
   - Prefer simpler models (Ridge) when performance is close, to reduce overfitting and improve robustness.
   - Select Hybrid only when it provides meaningful improvement.

**When Auto is recommended**
- Users without ML background (default case)
- Mixed-objective problems
- Early-stage projects with limited data
- When you prefer robustness over aggressive fitting

#### Manual (advanced)
User overrides the strategy per objective (Ridge, RF, Hybrid, or Auto per objective).

**When Manual is recommended**
- You already validated the best model family for a certain objective type
- You need consistent internal standards across projects
- You are diagnosing model/data issues

---

### 6.3 Practical recommendations (rules of thumb)

PAMO supports three model strategies per objective:

- **Poly(deg=2) + Ridge** (“Ridge”)
- **Random Forest Regression** (“RF”)
- **Hybrid** = Ridge baseline + RF residual corrector (“Hybrid”)

Because PAMO is multi-objective, **each objective may require a different strategy** (e.g., Heating may be smooth, while Comfort may be threshold-driven).

### Default recommendation
Use **Auto** unless there is a specific reason to override.

Auto evaluates the strategies **per objective** using cross-validation and selects the most reliable one.

---

### When to choose each strategy (manual override)

#### Choose **Ridge** when you want robustness and smooth behavior
Use Ridge if any of the following applies:

- Dataset is **small** (roughly **< 80–150 samples** for the given objective).
- Objective is expected to be **smooth / monotonic-ish** (e.g., heating load vs insulation level).
- You want **stable optimization** with “physically plausible” interpolation.
- You expect **some extrapolation** beyond the training domain (Ridge typically fails more gracefully than RF).

**Ridge warning sign:** if learning curves show a persistent Training–CV gap and CV performance stalls, the objective may require nonlinear correction (Hybrid/RF).

---

#### Choose **Random Forest** when you expect thresholds and strong nonlinearity (and you have enough data)
Use RF if:

- You have **strong coverage** of the parameter space (roughly **> 150–300 samples**, or a well-designed DOE).
- The objective has **regime changes / thresholds** (controls, shading logic, night ventilation, comfort metrics).
- You operate mostly **inside** the training domain (RF is primarily an interpolator; extrapolation is weak).

**RF warning sign:** large Training–CV gap or unstable CV across folds → risk of overfitting. Increase RF regularization (e.g., higher `min_samples_leaf`, cap `max_depth`) or prefer Hybrid.

---

#### Choose **Hybrid** when you want both stability and nonlinearity
Use Hybrid when:

- The objective has a strong global trend **plus** local nonlinear effects.
- Ridge is “close” but misses important regimes; RF alone looks unstable.
- You want a strong default for **mixed physics + control logic** behavior.

Hybrid is often appropriate for:
- thermal comfort / overheating metrics
- daylight metrics influenced by shading/geometry thresholds
- operational KPIs affected by control logic

---

### Auto selection rule used by PAMO (simple and robust)

For each objective, PAMO computes cross-validated error for **Ridge**, **RF**, and **Hybrid** (e.g., using NMAE).

Selection logic:

1. Start with **Ridge** as baseline.
2. If **Hybrid** improves CV error by **≥ 5%** vs Ridge → select **Hybrid**.
3. Else if **RF** improves CV error by **≥ 8%** vs Ridge → select **RF**.
4. Otherwise → keep **Ridge**.

Rationale:
- Hybrid is usually the safer “complex” option, so it needs only modest improvement to justify selection.
- RF can be less stable and weaker at extrapolation, so it should win by a larger margin before selection.

---

### Safety guardrail (recommended)
Even if RF/Hybrid wins on CV error, fall back to Ridge if the model shows a strong overfitting signature:

- **Training R² − CV R² > 0.08** (persistent at larger training sizes)

This guardrail helps prevent selecting overly flexible models when the dataset is small, noisy, or inconsistent.

---

### One-sentence guide for non-ML users
- **Ridge** = stable and smooth, best when data is limited.  
- **RF** = powerful for nonlinear thresholds, best when data coverage is strong.  
- **Hybrid** = best general-purpose choice when reality is mixed (trend + nonlinear regimes).  


---

## 7. Precision, reliability, and overfitting

### 7.1 What "precision" means here
Surrogate accuracy is bounded by:
- dataset quality and size
- coverage of the parameter space
- noise / inconsistency across simulation setups

Recommended workflow:
1. Use PAMO to narrow the design space.
2. Re-simulate the top candidates with the full physics engine (IESVE, etc.) for confirmation.

### 7.2 Overfitting in PAMO context
Overfitting is when a model learns noise/quirks in the training data and fails on new variants. Cross-validation is used to reduce this risk, and Auto strategy prefers simpler models when performance is comparable.

---

## 8. Setup controls (sidebar)

### Number of Trees in Random Forest (1-200)
Controls RF complexity and the RF component in Hybrid.

Guidance:
- 50-120 is often a good range.
- More trees reduces variance but increases runtime.

### Cross-Validation Folds (3-20)
Controls stability of evaluation:
- 5 folds for fast iteration
- 10 folds for more reliable selection
- 15-20 folds for maximum robustness (slower)

### Number of CPU cores
Controls parallel computation (where applicable). Higher values can increase memory usage.

---

## 9. Learning curves

Learning curves show how model error changes with training set size and help answer:
- Do we need more data?
- Is the model underfitting or overfitting?

Typical interpretations:
- Training error low, validation error high: overfitting (more data / more regularization)
- Both errors high: underfitting (add drivers/features, consider RF/Hybrid)
- Validation error decreases steadily with more data: collecting more data likely helps

---

## 10. Caching behavior (cached version)

This version caches trained models to improve responsiveness.

Cache invalidation is tied to:
- a signature of the uploaded dataset
- key hyperparameters (folds, RF trees, cores, etc.)

If you suspect stale behavior:
- re-upload the training dataset (or modify one cell and re-upload)
- retrain and confirm metrics update

---

## 11. Troubleshooting

### "No trained models available"
- Train models first in the Training tab.
- Confirm parameter and objective column names match your dataset.

### Optimization slow
- Reduce parameter search space (ranges/step sizes)
- Reduce CV folds while iterating
- Reduce RF trees if RF/Hybrid dominates runtime

### Precision poor
Usually due to:
- insufficient training samples
- inconsistent simulation assumptions
- missing drivers as inputs (e.g., system mode, schedule category)
- sparse coverage around critical thresholds/regimes

Remediation:
- add targeted samples in poorly performing regions
- include missing drivers as explicit parameters
- rely on Auto to choose RF/Hybrid for nonlinear objectives

---

## 12. Best practices for consulting use

### Design-stage (new buildings)
- Sample parameters systematically (Latin hypercube / structured grid).
- Use PAMO to identify Pareto candidates quickly.
- Re-simulate top 5-20 candidates for confirmation.

### Operation-stage optimization
- Train on operational data + drivers (weather, occupancy proxies).
- Use PAMO to propose setpoint/control adjustments.
- Validate with short-term tests or digital twin simulation.

### Market / business intelligence
- Aggregate anonymized project outcomes with consistent metadata.
- Identify parameter patterns linked to performance targets.
- Support early-stage feasibility and benchmarking.

---

## 13. Governance and data strategy (for scaling)

To scale PAMO into a company capability:
- standardize a minimum dataset schema
- store modeling assumptions as explicit metadata
- version datasets and tool releases
- implement validation gates: surrogate suggests -> simulation verifies

---

## 14. Model strategy quick cheat sheet

**Default:** Auto

If you must choose manually:
- **Ridge:** smooth, stable, smaller datasets, "physics-like" trends
- **RF:** complex nonlinearity, thresholds, strong interactions, good coverage
- **Hybrid:** mix of both; Ridge close but misses nonlinear regimes

---

**Document generated:** 2025-12-28
