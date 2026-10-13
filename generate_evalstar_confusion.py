import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report
)


# ============================================================
# Paths
# ============================================================

TRAIN_FILE = "df_posterior_probabilities/train_df_softmax.xlsx"
EVAL_STAR_FILE = "df_posterior_probabilities/eval_star_df_softmax.xlsx"

OUTPUT_DIR = "results/attribution"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load training data
# ============================================================

train_df = pd.read_excel(TRAIN_FILE)

# Only spoof samples
train_df = train_df[
    train_df["target"] == "spoof"
].copy()

X_train = train_df.iloc[:, 3:].values
y_train = train_df["spoof_attack"].values


# ============================================================
# Load Eval* data
# ============================================================

eval_df = pd.read_excel(EVAL_STAR_FILE)

X_eval = eval_df.iloc[:, 3:].values
y_eval = eval_df["spoof_attack"].values


print("Training shape:", X_train.shape)
print("Eval* shape:", X_eval.shape)


# ============================================================
# Train the same tree used for Eval*
# ============================================================

clf = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

clf.fit(X_train, y_train)


# ============================================================
# Prediction
# ============================================================

predictions = clf.predict(X_eval)

accuracy = accuracy_score(y_eval, predictions)

print("\nEval* Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_eval,
        predictions,
        labels=["A04", "A06"],
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

labels = ["A04", "A06"]

cm = confusion_matrix(
    y_eval,
    predictions,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Eval* Attack Attribution Confusion Matrix"
)

plt.tight_layout()


# ============================================================
# Save
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "evalstar_confusion_matrix.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved:")
print(output_file)