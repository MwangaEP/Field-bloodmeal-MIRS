# =============================================================================
# Blood Meal Source Classification (Human vs Bovine)
# UMAP dimensionality reduction — unsupervised visualisation
#
# Goal: assess whether human and bovine blood spectra in mosquitoes remain
# separable at 6, 12, 24, and 48 hours post-feeding without using any labels
# during the embedding step (purely unsupervised).
#
# Global scaling is appropriate here because UMAP is used only for
# visualisation — there is no held-out test set that could be contaminated.
# =============================================================================

#%%
import os
from time import time
 
import numpy as np
import pandas as pd
from collections import Counter
 
import umap
 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
 
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.set(
    context="paper",
    style="white",
    palette="deep",
    font_scale=1.5,
    color_codes=True,
    rc={"font.family": "DejaVu Sans"},
)
plt.rcParams["figure.figsize"] = [6, 4]
 
# Fixed class order and colours — consistent with all other scripts
CLASS_NAMES = ["Bovine", "Human"]
CLASS_COLORS = {"Bovine": "#1f77b4", "Human": "#ff7f0e"}
 
HOURS       = ["6H", "12H", "24H", "48H"]
HOUR_TITLES = ["6 Hours", "12 Hours", "24 Hours", "48 Hours"]

savedir = os.path.join("..", "Results", "UMAP")
 
#%%
# =============================================================================
# 1. Load and prepare blood-meal hours data
# =============================================================================

blood_hours_df = pd.read_csv(
    os.path.join("..", "Data", "Bloodfed_hours.dat"), 
    delimiter = '\t'
)

# Rename host for clarity
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('CW', 'Bovine')
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('HN', 'Human')

blood_hours_df = blood_hours_df.drop(['Cat1', 'Cat2', 'StoTime'], axis = 1)

print("Hours distribution:     ", Counter(blood_hours_df["Cat4"]))
print("Blood meal distribution:", Counter(blood_hours_df["Cat3"]))

X      = blood_hours_df.drop(["Cat3", "Cat4"], axis=1)
y_host = blood_hours_df["Cat3"]   # Bovine / Human — used to colour points
y_hour = blood_hours_df["Cat4"]   # 6H / 12H / 24H / 48H — used to split panels

# Global scaling: all samples scaled together so UMAP embeddings are
# comparable across hours within the same 2-D space.
X_scaled = StandardScaler().fit_transform(np.asarray(X))

#%%
# =============================================================================
# 2. Helper: 2 × 2 UMAP panel plot
# =============================================================================

def plot_umap_panels(X_umap, y_host, y_hour, title=None, save_path=None):
    """Plot UMAP embeddings split into one panel per time point.
 
    Points are coloured by host (Bovine / Human). The same embedding space
    is used for all panels so structure across hours is directly comparable.
 
    Args:
        X_umap:     (n_samples, 2) UMAP embedding array.
        y_host:     Series / array of host labels ('Bovine' / 'Human').
        y_hour:     Series / array of hour labels ('6H', '12H', ...).
        title:      Optional figure suptitle.
        save_path:  If provided, saves the figure to this path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    axes = axes.flatten()
 
    for ax, hour, hour_title in zip(axes, HOURS, HOUR_TITLES):
        hour_mask = np.asarray(y_hour == hour)
 
        for host in CLASS_NAMES:                          # fixed order, fixed colours
            host_mask = hour_mask & np.asarray(y_host == host)
            ax.scatter(
                X_umap[host_mask, 0],
                X_umap[host_mask, 1],
                label=host,
                color=CLASS_COLORS[host],
                alpha=0.7,
                s=18,
            )
 
        ax.set_title(hour_title, fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
 
    # Single shared legend above all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
 
    if title:
        fig.suptitle(title, y=1.01, fontweight="bold")
 
    plt.tight_layout(rect=[0, 0, 1, 0.95])
 
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        print(f"Saved → {save_path}")
 
    plt.close(fig)

# =============================================================================
# 3. Exploratory UMAP with initial parameters
# =============================================================================
 
umap_model_init = umap.UMAP(
    n_neighbors=32,
    min_dist=0.7,
    n_components=2,
    metric="chebyshev",
    random_state=42,
)
X_umap_init = umap_model_init.fit_transform(X_scaled)
 
plot_umap_panels(
    X_umap_init, y_host, y_hour,
    # title="Exploratory UMAP",
    save_path=os.path.join(savedir, "exploratory_umap.png")
)

#%%
# =============================================================================
# 4. Hyperparameter grid search — maximise host silhouette score
#
# Silhouette score measures how well Bovine and Human clusters are separated
# in the 2-D embedding. We use y_host (not y_hour) because the scientific
# question is whether host identity drives the spectral structure.
# =============================================================================
 
neighbors_grid = [5, 10, 15, 20, 30]
min_dist_grid  = [0.0, 0.05, 0.1, 0.2, 0.5]
metrics_grid   = ["euclidean", "cosine", "correlation"]
 
results = []
start_time = time()
 
for umap_metric in metrics_grid:
    for n_neighbors in neighbors_grid:
        for min_dist in min_dist_grid:
 
            embedding = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
                metric=umap_metric,
                random_state=42,
            ).fit_transform(X_scaled)
 
            # Silhouette w.r.t. host identity: score → 1 means tight,
            # well-separated Bovine / Human clusters in the embedding.
            sil = silhouette_score(embedding, y_host)
 
            results.append({
                "metric":     umap_metric,
                "n_neighbors": n_neighbors,
                "min_dist":   min_dist,
                "silhouette": sil,
            })
 
elapsed = time() - start_time
print(f"\nGrid search complete. Time: {elapsed:.1f} s ({elapsed / 60:.2f} min)")
 
results_df = (
    pd.DataFrame(results)
    .sort_values(by="silhouette", ascending=False)
    .reset_index(drop=True)
)
 
print(results_df.head(10).to_string(index=False))
 
# Save grid search results so the work is not lost if the kernel resets
results_df.to_csv(
    os.path.join(savedir, "umap_grid_search_results.csv"),
    index = False,
)

#%%
# =============================================================================
# 5. Final UMAP with best parameters from grid search
# =============================================================================
 
best = results_df.iloc[0]
print(f"\nBest parameters: metric={best['metric']}, "
      f"n_neighbors={int(best['n_neighbors'])}, "
      f"min_dist={best['min_dist']}, "
      f"silhouette={best['silhouette']:.4f}")
 
umap_model_best = umap.UMAP(
    n_neighbors=int(best["n_neighbors"]),
    min_dist=best["min_dist"],
    n_components=2,
    metric=best["metric"],
    random_state=42,
)
X_umap_best = umap_model_best.fit_transform(X_scaled)
 
plot_umap_panels(
    X_umap_best, y_host, y_hour,
    save_path=os.path.join(savedir, "final_umap.png"),
)

# %%
