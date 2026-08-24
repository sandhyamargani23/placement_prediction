import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET = "C:/Users/SANDHYA/PycharmProjects/placement_prediction/dataset/placement_predict_50K_Raw.csv"

OUTPUT_FOLDER = "C:/Users/SANDHYA/PycharmProjects/placement_prediction/outputs/Logistic_Regression_Binary_Classify_M2"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 2. LOAD PLACEMENT DATASET
# ============================================================

if not os.path.exists(DATASET):
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET}\n"
        "Please check the file path."
    )

df = pd.read_csv(DATASET)

print("\n================================================")
print("DATASET")
print("================================================")

print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

FEATURES = [
    "CGPA",
    "HistoryOfBacklogs",
    "Internships"
]

TARGET = "PlacementStatus"


# Check whether required columns exist
missing_columns = [
    column
    for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nRequired columns not found: {missing_columns}\n"
        f"Available columns are:\n{df.columns.tolist()}"
    )


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n================================================")
print("MISSING VALUE CHECK")
print("================================================")

print(df[FEATURES + [TARGET]].isnull().sum())


# Remove missing rows only if they exist
df_model = df[FEATURES + [TARGET]].dropna().copy()

print("\nOriginal Records:", len(df))
print("Records Used for Model:", len(df_model))


# ============================================================
# 5. SELECT X AND y
# ============================================================

X = df_model[FEATURES].values.astype(float)

y_raw = df_model[TARGET].values


# ============================================================
# 6. CONVERT TARGET TO BINARY 0/1
# ============================================================

unique_values = np.unique(y_raw)

print("\n================================================")
print("TARGET VALUES")
print("================================================")

print(unique_values)


# Case 1: Already 0 and 1
if set(unique_values) == {0, 1}:
    y = y_raw.astype(int)

# Case 2: Preprocessed/scaled target with two unique values
elif len(unique_values) == 2:

    negative_value = np.min(unique_values)
    positive_value = np.max(unique_values)

    y = np.where(
        y_raw == positive_value,
        1,
        0
    ).astype(int)

    print(
        "\nScaled target detected and converted to binary:"
    )

    print(
        f"{negative_value} -> 0 (Not Placed)"
    )

    print(
        f"{positive_value} -> 1 (Placed)"
    )

else:
    raise ValueError(
        "\nPlacementStatus must contain exactly two classes."
    )


print("\nFeatures:")
print(FEATURES)

print("\nTarget:")
print(TARGET)

print("\nTarget Distribution:")
print(pd.Series(y).value_counts())


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\n================================================")
print("TRAIN-TEST SPLIT")
print("================================================")

print("Training Records:", len(X_train))
print("Testing Records :", len(X_test))


# ============================================================
# 8. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. SIGMOID FUNCTION
# ============================================================

def sigmoid(z):
    """
    Sigmoid Function

    Converts linear score into probability:

             1
    -------------------
    1 + exp(-z)
    """

    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


# ============================================================
# 10. SIGMOID GRAPH
# ============================================================

z_values = np.linspace(-10, 10, 500)

sigmoid_values = sigmoid(z_values)

plt.figure(figsize=(8, 6))

plt.plot(
    z_values,
    sigmoid_values,
    color="blue",
    linewidth=3,
    label="Sigmoid Function"
)

plt.axhline(
    0.5,
    color="red",
    linestyle="--",
    label="Decision Threshold = 0.5"
)

plt.axvline(
    0,
    color="black",
    linestyle="--"
)

plt.xlabel("z")

plt.ylabel("Sigmoid(z)")

plt.title("Sigmoid Function")

plt.legend()

plt.grid(alpha=0.3)

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "01_sigmoid.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. CROSS-ENTROPY LOSS
# ============================================================

def cross_entropy_loss(
        y_true,
        y_probability
):

    epsilon = 1e-15

    y_probability = np.clip(
        y_probability,
        epsilon,
        1 - epsilon
    )

    loss = -np.mean(
        y_true * np.log(y_probability)
        +
        (1 - y_true) *
        np.log(1 - y_probability)
    )

    return loss


# ============================================================
# 12. INITIALIZE LOGISTIC REGRESSION
# ============================================================

number_of_features = X_train_scaled.shape[1]

weights = np.zeros(
    number_of_features
)

bias = 0.0

learning_rate = 0.05

epochs = 3000

loss_history = []


# ============================================================
# 13. TRAIN USING GRADIENT DESCENT
# ============================================================

print("\n================================================")
print("TRAINING LOGISTIC REGRESSION")
print("================================================")

m = len(y_train)

for epoch in range(epochs):

    # Linear model
    z = np.dot(
        X_train_scaled,
        weights
    ) + bias

    # Sigmoid probability
    probability = sigmoid(z)

    # Cross-entropy loss
    loss = cross_entropy_loss(
        y_train,
        probability
    )

    loss_history.append(loss)

    # Error
    error = probability - y_train

    # Gradient of weights
    dw = (
        1 / m
    ) * np.dot(
        X_train_scaled.T,
        error
    )

    # Gradient of bias
    db = (
        1 / m
    ) * np.sum(error)

    # Update weights
    weights -= (
        learning_rate * dw
    )

    # Update bias
    bias -= (
        learning_rate * db
    )

    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Loss: {loss:.6f}"
        )


print("\nModel trained successfully.")


# ============================================================
# 14. DISPLAY MODEL PARAMETERS
# ============================================================

print("\n================================================")
print("MODEL PARAMETERS")
print("================================================")

for feature, weight in zip(
        FEATURES,
        weights
):

    print(
        f"{feature}: {weight:.6f}"
    )

print(
    f"Bias: {bias:.6f}"
)


# ============================================================
# 15. CROSS-ENTROPY LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    range(1, epochs + 1),
    loss_history,
    color="purple",
    linewidth=2
)

plt.xlabel("Epoch")

plt.ylabel("Cross-Entropy Loss")

plt.title(
    "Logistic Regression - Cross-Entropy Loss"
)

plt.grid(alpha=0.3)

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "02_cross_entropy_loss.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 16. PREDICTION FUNCTIONS
# ============================================================

def predict_probability(X_input):

    z = np.dot(
        X_input,
        weights
    ) + bias

    return sigmoid(z)


def predict(
        X_input,
        threshold=0.5
):

    probability = predict_probability(
        X_input
    )

    return (
        probability >= threshold
    ).astype(int)


# ============================================================
# 17. TEST SET PREDICTION
# ============================================================

test_probability = predict_probability(
    X_test_scaled
)

y_pred = predict(
    X_test_scaled
)


# ============================================================
# 18. MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n================================================")
print("MODEL PERFORMANCE")
print("================================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Placed",
            "Placed"
        ],
        zero_division=0
    )
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    cmap="Blues"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    [
        "Not Placed",
        "Placed"
    ]
)

plt.yticks(
    [0, 1],
    [
        "Not Placed",
        "Placed"
    ]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")


for i in range(2):

    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=16
        )


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "03_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 20. DECISION BOUNDARY
# ============================================================

# The model uses:
#
# 1. CGPA
# 2. HistoryOfBacklogs
# 3. Internships
#
# A 2D visualization is created using:
#
# CGPA
# HistoryOfBacklogs
#
# Internships is fixed to its mean value.


# ============================================================
# CREATE CGPA AND BACKLOG GRID
# ============================================================

cgpa_values = np.linspace(
    df_model["CGPA"].min() - 0.2,
    df_model["CGPA"].max() + 0.2,
    300
)

backlog_values = np.linspace(
    df_model["HistoryOfBacklogs"].min() - 1,
    df_model["HistoryOfBacklogs"].max() + 1,
    300
)

CGPA_grid, Backlogs_grid = np.meshgrid(
    cgpa_values,
    backlog_values
)


# Fix Internships at the mean value
internship_fixed_value = (
    df_model["Internships"].mean()
)

INTERNSHIPS_grid = np.full_like(
    CGPA_grid,
    internship_fixed_value
)


# Create 3-feature grid
grid = np.column_stack([
    CGPA_grid.ravel(),
    Backlogs_grid.ravel(),
    INTERNSHIPS_grid.ravel()
])


# Standardize grid
grid_scaled = scaler.transform(
    grid
)


# Calculate probability
grid_probability = predict_probability(
    grid_scaled
)


grid_probability = grid_probability.reshape(
    CGPA_grid.shape
)


# ============================================================
# PLOT DECISION BOUNDARY
# ============================================================

plt.figure(figsize=(10, 7))


# Probability regions
contour = plt.contourf(
    CGPA_grid,
    Backlogs_grid,
    grid_probability,
    levels=50,
    cmap="RdYlGn",
    alpha=0.35
)

plt.colorbar(
    contour,
    label="Placement Probability"
)


# Decision boundary at probability = 0.5
plt.contour(
    CGPA_grid,
    Backlogs_grid,
    grid_probability,
    levels=[0.5],
    colors="black",
    linewidths=3
)


# Original Not Placed data
plt.scatter(
    df_model.loc[
        y == 0,
        "CGPA"
    ],
    df_model.loc[
        y == 0,
        "HistoryOfBacklogs"
    ],
    color="red",
    edgecolor="black",
    s=30,
    alpha=0.6,
    label="Not Placed"
)


# Original Placed data
plt.scatter(
    df_model.loc[
        y == 1,
        "CGPA"
    ],
    df_model.loc[
        y == 1,
        "HistoryOfBacklogs"
    ],
    color="green",
    edgecolor="black",
    s=30,
    alpha=0.6,
    label="Placed"
)


plt.xlabel("CGPA")

plt.ylabel("HistoryOfBacklogs")

plt.title(
    "Logistic Regression Decision Boundary"
)

plt.legend()

plt.grid(alpha=0.2)


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "04_decision_boundary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 21. PREDICT A NEW STUDENT
# ============================================================

# Feature order:
#
# CGPA
# HistoryOfBacklogs
# Internships

new_student = np.array([
    [
        8.0,  # CGPA
        0,    # HistoryOfBacklogs
        1     # Internships
    ]
])


# Standardize the new student
new_student_scaled = scaler.transform(
    new_student
)


# Calculate probability
new_probability = predict_probability(
    new_student_scaled
)[0]


# Calculate class
new_prediction = int(
    new_probability >= 0.5
)


print("\n================================================")
print("NEW STUDENT PREDICTION")
print("================================================")

print(
    "CGPA:",
    new_student[0][0]
)

print(
    "History of Backlogs:",
    int(new_student[0][1])
)

print(
    "Internships:",
    int(new_student[0][2])
)

print(
    "Placement Probability:",
    f"{new_probability * 100:.2f}%"
)


if new_prediction == 1:

    print(
        "Prediction: PLACED"
    )

else:

    print(
        "Prediction: NOT PLACED"
    )


# ============================================================
# 22. SAVE PREDICTION RESULTS
# ============================================================

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred,
    "Placement_Probability": test_probability
})


results_path = os.path.join(
    OUTPUT_FOLDER,
    "prediction_results.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 23. SAVE MODEL METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})


metrics_path = os.path.join(
    OUTPUT_FOLDER,
    "model_performance_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 24. FINISH
# ============================================================

print("\n================================================")
print("PROGRAM COMPLETED SUCCESSFULLY")
print("================================================")

print(
    "\nAll output files are stored in:"
)

print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)


print("\nGenerated files:")

print(
    "1. 01_sigmoid.png"
)

print(
    "2. 02_cross_entropy_loss.png"
)

print(
    "3. 03_confusion_matrix.png"
)

print(
    "4. 04_decision_boundary.png"
)

print(
    "5. prediction_results.csv"
)

print(
    "6. model_performance_metrics.csv"
)