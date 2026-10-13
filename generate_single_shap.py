import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Paths
# ============================================================

TRAIN_FILE = "df_posterior_probabilities/train_df_softmax.xlsx"
EVAL_FILE = "df_posterior_probabilities/eval_star_df_softmax.xlsx"

OUTPUT_DIR = "results/shap_single"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Attribute names
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
# Load training data
# ============================================================

train_df = pd.read_excel(TRAIN_FILE)

train_df = train_df[
    train_df["target"] == "spoof"
].copy()

X_train = train_df.iloc[:, 3:].values
y_train = train_df["spoof_attack"].values


# ============================================================
# Load Eval* data
# ============================================================

eval_df = pd.read_excel(EVAL_FILE)

X_eval = eval_df.iloc[:, 3:].values
y_eval = eval_df["spoof_attack"].values


# ============================================================
# Train frozen decision tree
# ============================================================

clf = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

clf.fit(X_train, y_train)


# ============================================================
# SHAP explainer
# ============================================================

explainer = shap.TreeExplainer(clf)


# ============================================================
# Select one A04 and one A06 sample
# ============================================================

a04_indices = np.where(y_eval == "A04")[0]
a06_indices = np.where(y_eval == "A06")[0]

a04_index = a04_indices[0]
a06_index = a06_indices[0]


# ============================================================
# Function for explaining one sample
# ============================================================

def explain_sample(index, expected_class, name):

    x = X_eval[index:index + 1]

    actual = y_eval[index]
    prediction = clf.predict(x)[0]

    print("\n======================================")
    print(name)
    print("======================================")

    print("Actual class:", actual)
    print("Predicted class:", prediction)

    # SHAP values
    shap_values = explainer.shap_values(x)

    print("SHAP shape:", shap_values.shape)

    class_index = list(clf.classes_).index(expected_class)

    # SHAP 0.49+ format:
    # samples × features × classes
    sample_shap = shap_values[0, :, class_index]

    # Rank attributes
    ranking = pd.DataFrame({
        "Feature": feature_names,
        "SHAP": sample_shap,
        "Absolute_SHAP": np.abs(sample_shap)
    })

    ranking = ranking.sort_values(
        "Absolute_SHAP",
        ascending=False
    )

    print("\nTop contributing attributes:")

    print(
        ranking.head(10).to_string(index=False)
    )

    # Save ranking
    ranking.to_excel(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_SHAP.xlsx"
        ),
        index=False
    )

    # Plot top 10
    top = ranking.head(10).sort_values("SHAP")

    plt.figure(figsize=(10, 6))

    plt.barh(
        top["Feature"],
        top["SHAP"]
    )

    plt.axvline(
        0,
        linewidth=1
    )

    plt.xlabel("SHAP value")
    plt.ylabel("Attribute")

    plt.title(
        f"{name}: SHAP Explanation\n"
        f"Actual={actual}, Predicted={prediction}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{name}_SHAP.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Explain A04
# ============================================================

explain_sample(
    a04_index,
    "A04",
    "A16_to_A04"
)


# ============================================================
# Explain A06
# ============================================================

explain_sample(
    a06_index,
    "A06",
    "A19_to_A06"
)


print("\n======================================")
print("SINGLE-SAMPLE SHAP COMPLETED")
print("======================================")