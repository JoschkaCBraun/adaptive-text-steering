import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# --- Define file paths ---
intermediate_dir = "/Users/joschka/Documents/0_Studium/0_ML_Master/0_current/masters_thesis/code/masters-thesis/data/0_paper_plots/intermediate_data"

# File from the first intermediate step (steerability metrics)
steerability_file = os.path.join(intermediate_dir, "intermediate_data_steerability_metrics_layer13_similarity_500.json")
# File from the second intermediate step (separation metrics)
separation_file = os.path.join(intermediate_dir, "separation_metrics_non_steered_llama2_7b_chat.json")
# File to store the merged data
merged_file = os.path.join(intermediate_dir, "merged_metrics.json")
# File to save the correlation matrix plot
plot_file = os.path.join(intermediate_dir, "merged_metrics_correlation_matrix.pdf")

# --- Step 1: Load JSON files ---
with open(steerability_file, 'r') as f:
    steerability_data = json.load(f)

with open(separation_file, 'r') as f:
    separation_data = json.load(f)

# --- Step 2: Merge the JSON files based on dataset names ---
# First, create dictionaries keyed by dataset name
steerability_dict = {entry["dataset"]: entry for entry in steerability_data}
separation_dict = {entry["dataset"]: entry for entry in separation_data}

# Check that both files have exactly the same dataset names
steerability_datasets = set(steerability_dict.keys())
separation_datasets = set(separation_dict.keys())

if steerability_datasets != separation_datasets:
    missing_in_sep = steerability_datasets - separation_datasets
    missing_in_steer = separation_datasets - steerability_datasets
    raise ValueError(f"Datasets mismatch!\nMissing in separation file: {missing_in_sep}\nMissing in steerability file: {missing_in_steer}")

# Merge the entries for each dataset
merged_data = []
for dataset in steerability_datasets:
    merged_entry = {}
    # From the first file:
    merged_entry["dataset"] = dataset
    merged_entry["anti_steerable_percentage"] = steerability_dict[dataset]["anti_steerable_percentage"]
    merged_entry["mean_logit_diff"] = steerability_dict[dataset]["mean_logit_diff"]
    merged_entry["layer13_similarity"] = steerability_dict[dataset]["layer13_similarity"]
    # From the second file:
    merged_entry["overlap_coef"] = separation_dict[dataset]["overlap_coef"]
    merged_entry["d_prime"] = separation_dict[dataset]["d_prime"]
    merged_entry["roc_auc"] = separation_dict[dataset]["roc_auc"]

    merged_data.append(merged_entry)

# --- Save the merged data ---
with open(merged_file, 'w') as f:
    json.dump(merged_data, f, indent=2)
print(f"Merged data saved to {merged_file}")

# --- Step 3: Compute a 6x6 correlation matrix across datasets ---
# The six measures:
measures = ["anti_steerable_percentage", "mean_logit_diff", "layer13_similarity",
            "overlap_coef", "d_prime", "roc_auc"]

# Create a pandas DataFrame from the merged data
df = pd.DataFrame(merged_data)

# Option 1: Use pandas built-in correlation method with Spearman
corr_matrix = df[measures].corr(method='spearman')
print("Spearman correlation matrix (pandas):")
print(corr_matrix)

# Option 2: If you want to verify using scipy.stats.spearmanr pairwise, you can do:
# (This code creates an empty 6x6 matrix and then fills it in with the pairwise Spearman correlation coefficient)
spearman_corr = np.zeros((len(measures), len(measures)))
p_values = np.zeros((len(measures), len(measures)))
for i, m1 in enumerate(measures):
    for j, m2 in enumerate(measures):
        rho, p_val = spearmanr(df[m1], df[m2])
        spearman_corr[i, j] = rho
        p_values[i, j] = p_val

print("\nSpearman correlation matrix (scipy):")
print(spearman_corr)

# --- Step 4: Plot the correlation matrix as a heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Spearman Correlation Matrix of Merged Metrics")
plt.tight_layout()
plt.savefig(plot_file, dpi=300)
plt.close()
print(f"Correlation matrix plot saved to {plot_file}")
