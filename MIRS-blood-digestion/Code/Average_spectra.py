#%%
# Import modules
import os
import numpy as np 
import pandas as pd
from collections import Counter 

import matplotlib.pyplot as plt # for making plots
import seaborn as sns

sns.set(context = "paper",
        style = "whitegrid",
        palette = "deep",
        font_scale = 2.0,
        color_codes = True,
        rc=None)
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# =============================================================================
# 3. Load and prepare blood-meal hours data
# =============================================================================

blood_hours_df = pd.read_csv(
    os.path.join("..", "Data", "Bloodfed_hours.dat"), 
    delimiter = '\t'
)

# Rename host names 
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('CW', 'Bovine')
blood_hours_df['Cat3'] = blood_hours_df['Cat3'].str.replace('HN', 'Human')

print("Hours distribution:", Counter(blood_hours_df['Cat4']))

FEATURE_COLS_TO_DROP = ["Cat1", "Cat2", "Cat3", "Cat4", "StoTime"]
 
TIME_POINTS = {
    "6H":  "6 hours",
    "12H": "12 hours",
    "24H": "24 hours",
    "48H": "48 hours",
}

# Build a dict of raw (unscaled) X and y per time point — scaling happens
# inside the helper functions so there is one clear place of responsibility.
raw_data = {}
for key, label in TIME_POINTS.items():
    subset = blood_hours_df[blood_hours_df["Cat4"] == key].copy()
    raw_data[key] = {
        "X": np.asarray(subset.drop(FEATURE_COLS_TO_DROP, axis=1)),
        "y": np.asarray(subset["Cat3"]),
        "label": label,
    }


#%%
# =============================================================================
# Average FTIR spectra — 2 × 2 panel plot
# Each panel shows mean ± SE for Bovine and Human at one time point.
# Uses raw_data dict and wavenumber columns built earlier in the pipeline.
# =============================================================================
 
# Wavenumber column names come from the original dataframe (after dropping
# the metadata columns). These are shared across all time points.
WAVENUMBERS = blood_hours_df.drop(FEATURE_COLS_TO_DROP, axis=1).columns.astype(float)
 
CLASS_COLORS = {"Bovine": "#1f77b4", "Human": "#ff7f0e"}
 
 
def compute_mean_se(X, y, class_name, wavenumbers):
    """Compute per-wavenumber mean and SE for one class.
 
    Args:
        X:           Raw feature array (n_samples, n_wavenumbers).
        y:           1-D string label array.
        class_name:  'Bovine' or 'Human'.
        wavenumbers: Float index of wavenumber column names.
 
    Returns:
        DataFrame with columns [Wavenumber, mean, se], sorted descending
        (4000 → 500 cm⁻¹ convention).
    """
    mask = y == class_name
    df   = pd.DataFrame(X[mask], columns=wavenumbers)
    return (
        pd.DataFrame({
            "Wavenumber": wavenumbers,
            "mean": df.mean(axis=0).values,
            "se":   df.sem(axis=0).values,
        })
        .sort_values("Wavenumber", ascending=False)
        .reset_index(drop=True)
    )
 
 
def plot_average_spectra(raw_data, wavenumbers, save_path=None):
    """Plot mean ± SE FTIR spectra for Bovine and Human in a 2 × 2 panel.
 
    Args:
        raw_data:    Dict keyed by time-point code ('6H', '12H', '24H', '48H'),
                     each value containing 'X' (array), 'y' (labels), 'label' (str).
        wavenumbers: Float array of wavenumber values (column names).
        save_path:   If provided, saves the figure to this path.
 
    Returns:
        fig: Matplotlib Figure object.
    """
    sns.set(
        context="paper",
        style="white",
        palette="deep",
        font_scale=1.5,
        color_codes=True,
        rc={"font.family": "DejaVu Sans"},
    )
 
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    axes = axes.flatten()
 
    for ax, (key, d) in zip(axes, raw_data.items()):
 
        for class_name, color in CLASS_COLORS.items():
            stats = compute_mean_se(d["X"], d["y"], class_name, wavenumbers)
 
            ax.plot(
                stats["Wavenumber"], stats["mean"],
                color=color, linewidth=0.7, label=class_name,
            )
            ax.fill_between(
                stats["Wavenumber"],
                stats["mean"] - stats["se"],
                stats["mean"] + stats["se"],
                color=color, alpha=0.25,
            )
 
        ax.set_title(d["label"], fontweight="bold")
        ax.set_xlabel("Wavenumbers / cm⁻¹", fontweight="bold")
        ax.set_ylabel("Absorbance", fontweight="bold")
        ax.set_xlim(max(wavenumbers), 500)  # 4000 → 500 convention
 
    # Single shared legend outside the panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Blood meal (mean ± SE)",
        loc="upper center",
        ncol=2,
        frameon=False,
    )
 
    plt.tight_layout(rect=[0, 0, 1, 0.93])
 
    if save_path:
        fig.savefig(save_path, dpi=500, bbox_inches="tight")
        print(f"Saved → {save_path}")
 
    return fig
 
 
# =============================================================================
# Call and save
# =============================================================================
 
fig = plot_average_spectra(
    raw_data,
    WAVENUMBERS,
    save_path=os.path.join("..", "Results", "average_spectra_blood_hours.png"),
)
plt.close(fig)

# %%
