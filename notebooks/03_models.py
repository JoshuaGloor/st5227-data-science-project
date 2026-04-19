# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python (st5227)
#     language: python
#     name: st5227
# ---

# %% [markdown]
# # 03 Modeling
#
# We predict the four target volumes (`wd_am`, `wd_pm`, `wd_midday`, `h_avg`) from the engineered features constructed in `02_features.py`.
#
# This notebook covers:
# 1. **Unsupervised**: cluster stops on the four targets to uncover a stop typology (commuter hub, weekend destination, etc.).
# 2. **Supervised**: compare four regressors (Linear, Ridge, Random Forest, Gradient Boosting) per target with 5-fold CV. Hyperparameters are tuned inside each fold where it matters. Paired t-tests assess whether score differences are statistically significant.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.data import load_bus_stops

RANDOM_STATE = 42

# %% [markdown]
# ## 1 Load features and targets

# %%
data_dir = Path.cwd().parent / "data" if "__file__" not in globals() else Path(__file__).resolve().parents[1] / "data"

features = pd.read_csv(data_dir / "features.csv", index_col="stop_id")
targets = pd.read_csv(data_dir / "targets.csv", index_col="stop_id")
bus_stops_coords = load_bus_stops().set_index("stop_id")

TARGET_NAMES = ["wd_am", "wd_pm", "wd_midday", "h_avg"]
FEATURE_NAMES = list(features.columns)

print(f"Features: {features.shape}, Targets: {targets.shape}")
print(f"Feature columns: {FEATURE_NAMES}")

# %% [markdown]
# ## 2 Unsupervised: stop typology via K-means on targets
#
# Before modeling, we cluster stops on their four target values to uncover archetypal usage patterns. Based on the distinct regimes observed in the violin plots - weekday commuter peaks, midday steady-state, and weekend/holiday patterns - we expect roughly four archetypal stop types:
# 1. **Commuter hubs**: high weekday rush, modest otherwise.
# 2. **All-day busy stops**: consistently high across all time buckets.
# 3. **Weekend-leaning stops**: weekend/holiday activity competitive with or exceeding weekday activity.
# 4. **Quiet stops**: low ridership across all periods.
#
# We therefore fit K-means with `k=4`.

# %%
K = 4

# Normalize each row to a profile: fractions summing to 1
# We did that because otherwise the cluster seemed to follow busyness instead of what we wanted to model.
target_profiles = targets[TARGET_NAMES].div(targets[TARGET_NAMES].sum(axis=1), axis=0)
# Drop rows where the sum was 0 (totally silent stops)
target_profiles = target_profiles.dropna()

scaler_targets = StandardScaler()
targets_scaled = scaler_targets.fit_transform(target_profiles)

kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(targets_scaled)
targets["cluster"] = cluster_labels

# Centroids in original (unscaled) target space for interpretation
centroids = pd.DataFrame(
    scaler_targets.inverse_transform(kmeans.cluster_centers_),
    columns=TARGET_NAMES,
).round(1)
centroids["n_stops"] = targets["cluster"].value_counts().sort_index().values
centroids

# %% [markdown]
# ### Cluster profiles; bar chart

# %%
fig, ax = plt.subplots(figsize=(10, 5))
centroids[TARGET_NAMES].plot(kind="bar", ax=ax)
ax.set_xlabel("Cluster")
ax.set_ylabel("Mean volume")
ax.set_title("Cluster centroids across the four targets")
ax.legend(title="Target", loc="upper right")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Cluster map; geographic distribution

# %%
# Merge cluster labels with coordinates
plot_df = targets[["cluster"]].join(bus_stops_coords, how="inner")

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    plot_df["lng"],
    plot_df["lat"],
    c=plot_df["cluster"],
    cmap="tab10",
    s=8,
    alpha=0.7,
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Bus stops colored by cluster")
ax.set_aspect("equal")
legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="lower right")
ax.add_artist(legend1)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Drop cluster column before continuing with regression
targets = targets.drop(columns="cluster")

# %% [markdown]
# ### Cluster interpretation
#
# Four archetypes emerge (slightly different than we hoped for) after normalizing each stop's target vector into a profile (fraction of ridership in each time bucket):
#
# | Cluster | Label | Profile | Likely type |
# |---------|-------|---------|-------------|
# | 0 | Mixed, morning-leaning | AM>PM>midday>weekend | General residential |
# | 1 | Strong morning origin | Dominant AM, near-zero weekend | Residential dorm/estate |
# | 2 | Strong evening destination | Dominant PM, near-zero weekend | Workplace/CBD |
# | 3 | Balanced all-weekday | Even AM/PM/midday, low weekend | Mixed-use / institutional |
#
# No cluster is dominated by weekend ridership, weekend activity is uniformly modest across all archetypes, meaning temporal variation is primarily a weekday story.
#
# **Geographic pattern**: Clusters 1 and 2 concentrate in Singapore's western region (Jurong and surroundings), consistent with the residential-industrial commute flows characteristic of the area. The rest of the network is dominated by the more balanced clusters 0 and 3.

# %% [markdown]
# ## 3 Supervised: regression on the four targets
#
# We use a log1p transform on each target because the violin plots showed heavy right-skew, otherwise models might suffer under this heavy difference in magnitude. Predictions are back-transformed with `expm1` before computing metrics.


# %%
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(model, X, y, cv):
    """Run CV, returning per-fold RMSE and MAE on the original (un-logged) scale."""

    rmse_scores, mae_scores = [], []
    for train_idx, test_idx in cv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Log-transform target for training
        model.fit(X_tr, np.log1p(y_tr))
        y_pred = np.expm1(model.predict(X_te))
        y_pred = np.clip(y_pred, 0, None)  # avoid negatives from numerical noise

        rmse_scores.append(rmse(y_te, y_pred))
        mae_scores.append(mean_absolute_error(y_te, y_pred))
    return np.array(rmse_scores), np.array(mae_scores)


# %% [markdown]
# ### Model definitions
#
# - **Linear**: ordinary least squares. No hyperparameters.
# - **Ridge**: L2-regularized linear. `alpha` tuned via `RidgeCV` (built-in LOO CV).
# - **Random Forest**: `min_samples_leaf` tuned via inner 3-fold CV. `n_estimators` fixed at 200.
# - **Gradient Boosting**: `learning_rate` and `max_depth` tuned via inner 3-fold CV.
#
# All models receive standardized features. We wrap them in a `Pipeline` so the scaler is re-fit per fold and doesn't leak test statistics.
#
# We do not explore more models and further tuning out of time constraints.


# %%
def make_models():
    """Construct fresh model instances. Called per fold/target to avoid state leak."""
    return {
        "Linear": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-2, 3, 20))),
            ]
        ),
        "RandomForest": GridSearchCV(
            RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            param_grid={
                "min_samples_leaf": [5, 10, 20],
            },
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        ),
        "GradBoost": GridSearchCV(
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
            param_grid={
                "learning_rate": [0.05, 0.1],
                "max_depth": [None, 5, 8],
            },
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        ),
    }


# %% [markdown]
# ### Run CV across all models and targets

# %%

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Store per-fold scores: {target: {model_name: array of rmse over folds}}
rmse_results = {}
mae_results = {}

for target in tqdm(TARGET_NAMES, desc="Targets"):
    rmse_results[target] = {}
    mae_results[target] = {}
    y = targets[target]

    models = make_models()
    for model_name, model in tqdm(models.items(), desc=f"Models ({target})", leave=False, total=len(models)):
        rmses, maes = evaluate_model(model, features, y, cv)
        rmse_results[target][model_name] = rmses
        mae_results[target][model_name] = maes
        tqdm.write(
            f"  [{target}] {model_name:13s}  "
            f"RMSE: {rmses.mean():7.2f} ± {rmses.std():6.2f}   "
            f"MAE: {maes.mean():7.2f} ± {maes.std():6.2f}"
        )

# %% [markdown]
# ### Summary table

# %%
summary_rows = []
for target in TARGET_NAMES:
    for model_name in rmse_results[target]:
        rmses = rmse_results[target][model_name]
        maes = mae_results[target][model_name]
        summary_rows.append(
            {
                "target": target,
                "model": model_name,
                "rmse_mean": rmses.mean(),
                "rmse_std": rmses.std(),
                "mae_mean": maes.mean(),
                "mae_std": maes.std(),
            }
        )
summary = pd.DataFrame(summary_rows)
summary_pivot = summary.pivot(index="model", columns="target", values="rmse_mean").round(1)
summary_pivot

# %% [markdown]
# ### Visualize per-fold RMSE across models

# %%
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, target in zip(axes, TARGET_NAMES):
    data = pd.DataFrame(rmse_results[target])
    sns.boxplot(data=data, ax=ax)
    ax.set_title(target)
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4 Statistical significance: paired t-tests
#
# For each target we test whether the best model's per-fold RMSE is significantly lower than each other model's. A paired t-test is used because both models are evaluated on the same CV folds, pairing removes fold-level variance (some folds contain harder test stops than others) and directly compares the per-fold difference in RMSE. A small p-value indicates that one model consistently outperforms the other across folds, rather than winning by luck on particular splits.

# %%
sig_rows = []
for target in TARGET_NAMES:
    scores = rmse_results[target]
    best_model = min(scores, key=lambda m: scores[m].mean())
    for other_model in scores:
        if other_model == best_model:
            continue
        stat, pval = stats.ttest_rel(scores[best_model], scores[other_model])
        sig_rows.append(
            {
                "target": target,
                "best_model": best_model,
                "other_model": other_model,
                "rmse_diff": scores[other_model].mean() - scores[best_model].mean(),
                "p_value": pval,
                "significant (p<0.05)": pval < 0.05,
            }
        )
sig_df = pd.DataFrame(sig_rows)
sig_df.round(4)

# %% [markdown]
# ## 5 Key findings
#
# 1. **GradBoost wins on every target**, with RandomForest close behind. The ranking is GradBoost -> RF -> Ridge -> Linear as we might expect and it is stable across all four targets.
#
# 2. **Ridge barely helps over plain Linear.** The regularization gain is very small across targets, suggesting feature multicollinearity is not a dominant issue. Both linear models are dominated by the tree-based models on this problem.
#
# 3. **Statistical significance is limited by fold-level variance.** The large gaps between linear and tree models are not significant at p<0.05. This is despite differences of 1500-2300 RMSE on average, because the linear models' per-fold RMSE is highly variable (see the wide boxplots). GradBoost vs RandomForest differences are small but consistent enough to be significant on `wd_am` (p=0.003) and `h_avg` (p=0.027).
#
# 4. **Clustering (unsupervised) reveals four commute archetypes**, not a busyness hierarchy: strong morning-origin stops, strong evening-destination stops, balanced all-weekday stops, and mixed morning-leaning stops. Clusters 1 and 2 (directional commute stops) concentrate in Singapore's western region (Jurong), consistent with its residential-industrial structure.
#
# ### Main takeaway
#
# Tree-based models (gradient boosting in particular) substantially outperform linear baselines on every target, cutting RMSE roughly in half. This indicates that bus-stop busyness depends on nonlinear interactions between the spatial features (e.g., the combined effect of HDB density *and* MRT proximity *and* POI counts) rather than on any single feature in isolation.
