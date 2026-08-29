import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================
# 1. SET PATHS
# =============================================================

dataset_path = "C:/Users/SANDHYA/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv"

output_dir = "C:/Users/SANDHYA/PycharmProjects/placement_prediction/outputs/EDA_Analysis_outputs"


# Create output directory
os.makedirs(output_dir, exist_ok=True)


# =============================================================
# 2. LOAD DATASET
# =============================================================

df = pd.read_csv(dataset_path)

print("==============================================")
print("Dataset Loaded Successfully")
print("==============================================")
print("Dataset Shape:", df.shape)
print("\nColumn Names:")
print(df.columns.tolist())


# =============================================================
# 3. SELECT NUMERICAL COLUMNS
# =============================================================

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\n==============================================")
print("Numerical Columns")
print("==============================================")
print(numerical_cols)


# =============================================================
# 4. CORRELATION MATRIX
# =============================================================

corr_matrix = df[numerical_cols].corr()

print("\n==============================================")
print("Correlation Matrix")
print("==============================================")
print(corr_matrix)


# =============================================================
# 5. CORRELATION HEATMAP
# =============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap of Numerical Features",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()


# Save heatmap
heatmap_path = os.path.join(
    output_dir,
    "correlation_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nCorrelation heatmap exported to:")
print(heatmap_path)


# =============================================================
# 6. BOXPLOTS: NUMERICAL FEATURES VS PLACEMENT STATUS
# =============================================================

target_col = "PlacementStatus"


if target_col in df.columns:

    print("\n==============================================")
    print("Generating Boxplots")
    print("==============================================")

    for col in numerical_cols:

        # Do not create a boxplot of PlacementStatus against itself
        if col == target_col:
            continue

        plt.figure(figsize=(6, 5))

        sns.boxplot(
            x=target_col,
            y=col,
            data=df,
            hue=target_col,
            palette="Set2",
            legend=False
        )

        plt.title(
            f"{col} vs {target_col}",
            fontsize=12,
            fontweight="bold"
        )

        plt.xlabel(target_col)
        plt.ylabel(col)

        plt.tight_layout()


        # File name
        boxplot_filename = (
            f"boxplot_{col}_vs_{target_col}.png"
        )

        boxplot_path = os.path.join(
            output_dir,
            boxplot_filename
        )


        # Save boxplot
        plt.savefig(
            boxplot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Exported: {boxplot_filename}")


else:

    print("\nWARNING:")
    print(
        f"Target column '{target_col}' "
        "was not found in the dataset."
    )

    print("\nAvailable columns are:")
    print(df.columns.tolist())


# =============================================================
# 7. COMPLETION MESSAGE
# =============================================================

print("\n==============================================")
print("ALL EDA TASKS COMPLETED SUCCESSFULLY!")
print("==============================================")

print("\nOutput files are saved in:")
print(output_dir)