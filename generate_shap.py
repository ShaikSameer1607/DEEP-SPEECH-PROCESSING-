import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier


# ============================================================
# 1. Paths
# ============================================================

TRAIN_FILE = "df_posterior_probabilities/train_df_softmax.xlsx"
EVAL_STAR_FILE = "df_posterior_probabilities/eval_star_df_softmax.xlsx"

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

# Only spoof samples are used for attack attribution
train_df = train_df[train_df["target"] == "spoof"].copy()

X_train = train_df.iloc[:, 3:].values
y_train = train_df["spoof_attack"].values


# ============================================================
# 4. Load Eval* data
# ============================================================

print("Loading Eval* data...")

eval_df = pd.read_excel(EVAL_STAR_FILE)

X_eval = eval_df.iloc[:, 3:].values
y_eval = eval_df["spoof_attack"].values


print("Training shape:", X_train.shape)
print("Eval* shape:", X_eval.shape)


# ============================================================
# 5. Train the SAME decision tree
# ============================================================

print("\nTraining decision tree...")

clf = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

clf.fit(X_train, y_train)

accuracy = clf.score(X_eval, y_eval)

print("Eval* accuracy:", accuracy)


# ============================================================
# 6. SHAP Tree Explainer
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(clf)

shap_values = explainer.shap_values(X_eval)

print("SHAP calculation completed.")


# ============================================================
# 7. Handle SHAP output format
# ============================================================

print("SHAP output type:", type(shap_values))

if isinstance(shap_values, list):

    # Older / multiclass format
    print("Number of SHAP arrays:", len(shap_values))

    # Use contribution toward A06
    class_names = list(clf.classes_)

    print("Classes:", class_names)

    if "A06" in class_names:
        class_index = class_names.index("A06")
    else:
        class_index = 0

    shap_for_plot = shap_values[class_index]

else:

    # New SHAP format
    print("SHAP array shape:", shap_values.shape)

    if len(shap_values.shape) == 3:

        class_names = list(clf.classes_)

        if "A06" in class_names:
            class_index = class_names.index("A06")
        else:
            class_index = 0

        shap_for_plot = shap_values[:, :, class_index]

    else:

        shap_for_plot = shap_values


# ============================================================
# 8. Create SHAP dataframe
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_for_plot),
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

print("\n===== SHAP FEATURE IMPORTANCE =====")

print(
    importance_df.to_string(index=False)
)


# ============================================================
# 9. Save importance table
# ============================================================

importance_file = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.xlsx"
)

importance_df.to_excel(
    importance_file,
    index=False
)

print("\nSaved:", importance_file)


# ============================================================
# 10. SHAP bar plot
# ============================================================

plt.figure(figsize=(12, 10))

shap.summary_plot(
    shap_for_plot,
    X_eval,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.tight_layout()

bar_plot = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.png"
)

plt.savefig(
    bar_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", bar_plot)


# ============================================================
# 11. SHAP beeswarm plot
# ============================================================

plt.figure(figsize=(12, 10))

shap.summary_plot(
    shap_for_plot,
    X_eval,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

beeswarm_plot = os.path.join(
    OUTPUT_DIR,
    "shap_beeswarm.png"
)

plt.savefig(
    beeswarm_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", beeswarm_plot)


print("\n======================================")
print("SHAP ANALYSIS COMPLETED")
print("======================================")