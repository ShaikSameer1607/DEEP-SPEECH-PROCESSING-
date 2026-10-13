import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_text


# ============================================================
# 1. Paths
# ============================================================

TRAIN_FILE = "df_posterior_probabilities/train_df_softmax.xlsx"
EVAL_FILE = "df_posterior_probabilities/eval_star_df_softmax.xlsx"

OUTPUT_DIR = "SHAP-plots/OurResults"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Attribute names
# ============================================================

attribute_names = [
    "Text",
    "Speech (human)",

    "NLP",
    "WORLD",
    "LPCC-MFCC",

    "HMM",
    "FF",
    "None (Duration)",

    "AR-RNN",
    "FF",
    "CART",
    "VAE",
    "GMM-UBM",

    "VAE",
    "One-hot",
    "None (Speaker)",

    "MCC + F0",
    "MCC + F0 + BAP",
    "MFCC + F0",
    "MCC + F0 + AP",
    "LPC",

    "WaveNet",
    "WORLD",
    "Waveform Concatenation",
    "Spectral Filtering + OLA"
]

feature_names = [
    f"S{i:02d} - {name}"
    for i, name in enumerate(attribute_names, start=1)
]


# ============================================================
# 3. Load training data
# ============================================================

print("Loading training data...")

train_df = pd.read_excel(TRAIN_FILE)

# Only spoof samples
train_df = train_df[
    train_df["target"] == "spoof"
].copy()

# Only A04 and A06
train_df = train_df[
    train_df["spoof_attack"].isin(["A04", "A06"])
].copy()

X_train = train_df.iloc[:, 3:].values
y_train = train_df["spoof_attack"].values


# ============================================================
# 4. Load Eval* data
# ============================================================

print("Loading Eval* data...")

eval_df = pd.read_excel(EVAL_FILE)

eval_df = eval_df[
    eval_df["spoof_attack"].isin(["A04", "A06"])
].copy()

X_eval = eval_df.iloc[:, 3:].values
y_eval = eval_df["spoof_attack"].values


print("\nTraining shape:", X_train.shape)
print("Eval* shape:", X_eval.shape)


# ============================================================
# 5. Train decision tree
# ============================================================

clf = DecisionTreeClassifier(
    max_depth=5
)

clf.fit(X_train, y_train)

accuracy = clf.score(X_eval, y_eval)

print("\nEval* Accuracy:", accuracy)

print("\n===== DECISION TREE =====")

print(
    export_text(
        clf,
        feature_names=feature_names
    )
)


# ============================================================
# 6. SHAP
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(X_eval)

print("SHAP type:", type(shap_values))

if isinstance(shap_values, list):

    print("Number of classes:", len(shap_values))

    class_names = list(clf.classes_)

    print("Classes:", class_names)

    # A06 explanation
    a06_index = class_names.index("A06")

    shap_a06 = shap_values[a06_index]

else:

    print("SHAP shape:", shap_values.shape)

    class_names = list(clf.classes_)

    print("Classes:", class_names)

    # SHAP 0.49+
    # shape = samples × features × classes

    a06_index = class_names.index("A06")

    shap_a06 = shap_values[:, :, a06_index]


# ============================================================
# 7. Calculate feature importance
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_a06),
    axis=0
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)


print("\n===== SHAP IMPORTANCE =====")

print(
    importance_df.to_string(index=False)
)


# ============================================================
# 8. Save Excel
# ============================================================

importance_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "EvalStar_SHAP_importance.xlsx"
    ),
    index=False
)


# ============================================================
# 9. SHAP bar plot
# ============================================================

plt.figure(figsize=(12, 10))

shap.summary_plot(
    shap_a06,
    X_eval,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "EvalStar_SHAP_bar.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. SHAP beeswarm
# ============================================================

plt.figure(figsize=(12, 10))

shap.summary_plot(
    shap_a06,
    X_eval,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "EvalStar_SHAP_beeswarm.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\n======================================")
print("EVAL* SHAP ANALYSIS COMPLETED")
print("======================================")