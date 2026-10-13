"""
Decision Tree Classification
----------------------------

This script performs:

1. Bonafide vs Spoof classification
2. Spoof Attack classification

The trained DecisionTreeClassifier models are also saved as:

decision_tree_results/
    bonafide_spoof_classification/
        decision_tree.pkl

    spoof_attack_classification/
        decision_tree.pkl

These .pkl files are used by live_test.py for live inference.
"""

import os
import argparse
import configparser
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

config = configparser.ConfigParser()

config.read(
    "config/emb_model_AASIST.conf"
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def plot_confusion_matrix(
    cm,
    classes,
    title="Confusion matrix"
):

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues
    )

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(
        len(classes)
    )

    plt.xticks(
        tick_marks,
        classes,
        rotation=45
    )

    plt.yticks(
        tick_marks,
        classes
    )

    threshold = cm.max() / 2.0

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                format(
                    cm[i, j],
                    "d"
                ),

                horizontalalignment="center",

                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )

    plt.tight_layout()

    plt.ylabel(
        "True label"
    )

    plt.xlabel(
        "Predicted label"
    )

    plt.grid(False)


# ============================================================
# DEVELOPMENT SET RESULTS
# ============================================================

def dev_data_results(
    X_dev,
    y_dev,
    df,
    dt_classifier,
    text_file_path,
    figure_dir,
    target_column
):

    predictions = (
        dt_classifier.predict(
            X_dev
        )
    )

    accuracy = accuracy_score(
        y_dev,
        predictions
    )

    with open(
        text_file_path,
        "a"
    ) as f:

        f.write(
            f"Development Set Testing Accuracy: "
            f"{accuracy}\n"
        )

        f.write(
            "-" * 50
            + "\n"
        )

        if target_column == "target":

            labels = list(
                df["target"].unique()
            )

            report = classification_report(
                y_dev,
                predictions,
                target_names=labels,
                zero_division=0
            )

            f.write(
                report
            )

        elif target_column == "spoof_attack":

            labels = sorted(
                list(
                    df["spoof_attack"].unique()
                )
            )

            report = classification_report(
                y_dev,
                predictions,
                labels=labels,
                zero_division=0
            )

            f.write(
                report
            )

        f.write(
            "-" * 50
            + "\n\n"
        )

    # --------------------------------------------------------
    # Development confusion matrix
    # --------------------------------------------------------

    if target_column == "target":

        label_list = list(
            df["target"].unique()
        )

    else:

        label_list = sorted(
            list(
                df["spoof_attack"].unique()
            )
        )

    cm = confusion_matrix(
        y_dev,
        predictions,
        labels=label_list
    )

    plt.figure()

    plot_confusion_matrix(
        cm,
        classes=label_list,
        title="Confusion matrix for Development Data"
    )

    os.makedirs(
        figure_dir,
        exist_ok=True
    )

    figure_path = os.path.join(
        figure_dir,
        "dev_confusion_matrix.png"
    )

    plt.savefig(
        figure_path,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# EVALUATION SET RESULTS
# ============================================================

def eval_data_results(
    X_eval,
    y_eval,
    df,
    df_eval,
    dt_classifier,
    text_file_path,
    figure_dir,
    task
):

    predictions = (
        dt_classifier.predict(
            X_eval
        )
    )

    with open(
        text_file_path,
        "a"
    ) as f:

        # ----------------------------------------------------
        # TASK 1
        # Bonafide vs Spoof
        # ----------------------------------------------------

        if task == 1:

            accuracy = accuracy_score(
                y_eval,
                predictions
            )

            f.write(
                f"Evaluation Set Testing Accuracy: "
                f"{accuracy}\n"
            )

            f.write(
                "-" * 50
                + "\n"
            )

            label_list = list(
                df["target"].unique()
            )

            cm = confusion_matrix(
                y_eval,
                predictions,
                labels=label_list
            )

            np.set_printoptions(
                precision=2
            )

            plt.figure()

            plot_confusion_matrix(
                cm,
                classes=label_list,
                title=(
                    "Confusion matrix "
                    "for Evaluation Data"
                )
            )

            report = classification_report(
                y_eval,
                predictions,
                labels=label_list,
                target_names=label_list,
                zero_division=0
            )

            f.write(
                report
            )

            f.write(
                "-" * 50
                + "\n\n"
            )

        # ----------------------------------------------------
        # TASK 2
        # Spoof attack classification
        # ----------------------------------------------------

        elif task == 2:

            label_list = list(
                df["spoof_attack"].unique()
            ) + list(
                df_eval["spoof_attack"].unique()
            )

            label_list = sorted(
                list(
                    set(label_list)
                )
            )

            accuracy = accuracy_score(
                y_eval,
                predictions
            )

            f.write(
                f"Evaluation Set Testing Accuracy: "
                f"{accuracy}\n"
            )

            f.write(
                "-" * 50
                + "\n"
            )

            report = classification_report(
                y_eval,
                predictions,
                labels=label_list,
                zero_division=0
            )

            f.write(
                report
            )

            f.write(
                "-" * 50
                + "\n\n"
            )

            cm = confusion_matrix(
                y_eval,
                predictions,
                labels=label_list
            )

            np.set_printoptions(
                precision=2
            )

            plt.figure(
                figsize=(20, 10)
            )

            plot_confusion_matrix(
                cm,
                classes=label_list,
                title=(
                    "Confusion matrix "
                    "for Evaluation Data"
                )
            )

            # ------------------------------------------------
            # Save evaluation confusion matrix
            # ------------------------------------------------

            os.makedirs(
                figure_dir,
                exist_ok=True
            )

            figure_path = os.path.join(
                figure_dir,
                "eval_confusion_matrix.png"
            )

            plt.savefig(
                figure_path,
                bbox_inches="tight"
            )

            plt.close()


# ============================================================
# BONAFIDE VS SPOOF CLASSIFICATION
# ============================================================

def bonafide_spoof(
    df,
    df_dev,
    df_eval,
    save_dir
):

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    save_dir = os.path.join(
        save_dir,
        "bonafide_spoof_classification"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Drop irrelevant attributes
    # --------------------------------------------------------

    df.drop(
        [
            "ID",
            "spoof_attack"
        ],
        axis=1,
        inplace=True
    )

    df_dev.drop(
        [
            "ID",
            "spoof_attack"
        ],
        axis=1,
        inplace=True
    )

    df_eval.drop(
        [
            "ID",
            "spoof_attack"
        ],
        axis=1,
        inplace=True
    )

    # ========================================================
    # PEARSON CORRELATION
    # ========================================================

    # --------------------------------------------------------
    # Training set
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 8)
    )

    corr = df.drop(
        ["target"],
        axis=1
    ).corr(
        method="pearson"
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues"
    )

    file_path = os.path.join(
        save_dir,
        "train_pearson_corr.png"
    )

    plt.savefig(
        file_path,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Development set
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 8)
    )

    corr = df_dev.drop(
        ["target"],
        axis=1
    ).corr(
        method="pearson"
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues"
    )

    file_path = os.path.join(
        save_dir,
        "dev_pearson_corr.png"
    )

    plt.savefig(
        file_path,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Evaluation set
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 8)
    )

    corr = df_eval.drop(
        ["target"],
        axis=1
    ).corr(
        method="pearson"
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues"
    )

    file_path = os.path.join(
        save_dir,
        "eval_pearson_corr.png"
    )

    plt.savefig(
        file_path,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # PREPARE DATA
    # ========================================================

    data = df.values

    X = data[:, 1:]
    y = data[:, 0]

    data_dev = df_dev.values

    X_dev = data_dev[:, 1:]
    y_dev = data_dev[:, 0]

    data_eval = df_eval.values

    X_eval = data_eval[:, 1:]
    y_eval = data_eval[:, 0]

    # ========================================================
    # DECISION TREE
    # ========================================================

    clf = DecisionTreeClassifier(
        max_depth=int(
            config[
                "decision_tree"
            ][
                "max_depth"
            ]
        )
    )

    clf = clf.fit(
        X,
        y
    )

    # ========================================================
    # SAVE TRAINED MODEL
    # ========================================================

    model_path = os.path.join(
        save_dir,
        "decision_tree.pkl"
    )

    joblib.dump(
        clf,
        model_path
    )

    print(
        "\nBonafide/Spoof decision tree saved to:"
    )

    print(
        model_path
    )

    # ========================================================
    # SAVE TREE FIGURE
    # ========================================================

    plt.figure(
        figsize=(20, 12)
    )

    plot_tree(
        clf,
        filled=True,
        feature_names=list(
            df.columns[1:]
        ),
        class_names=[
            str(c)
            for c in clf.classes_
        ],
        rounded=True
    )

    tree_path = os.path.join(
        save_dir,
        "decision_tree.png"
    )

    plt.savefig(
        tree_path,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # RESULTS FILE
    # ========================================================

    text_file_path = os.path.join(
        save_dir,
        "results.txt"
    )

    # Start a fresh result file
    with open(
        text_file_path,
        "w"
    ) as f:

        f.write(
            "BONAFIDE VS SPOOF CLASSIFICATION\n"
        )

        f.write(
            "=" * 50
            + "\n\n"
        )

    # ========================================================
    # DEVELOPMENT RESULTS
    # ========================================================

    dev_data_results(
        X_dev,
        y_dev,
        df,
        clf,
        text_file_path,
        save_dir,
        "target"
    )

    # ========================================================
    # EVALUATION RESULTS
    # ========================================================

    eval_data_results(
        X_eval,
        y_eval,
        df,
        df_eval,
        clf,
        text_file_path,
        save_dir,
        1
    )

    print(
        "Bonafide/Spoof classification complete."
    )


# ============================================================
# SPOOF ATTACK CLASSIFICATION
# ============================================================

def spoof_attack_classification(
    df,
    df_dev,
    df_eval,
    save_dir
):

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    save_dir = os.path.join(
        save_dir,
        "spoof_attack_classification"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ========================================================
    # DROP BONAFIDE RECORDS
    # ========================================================

    df = df[
        df["target"] != "bonafide"
    ].copy()

    df_dev = df_dev[
        df_dev["target"] != "bonafide"
    ].copy()

    df_eval = df_eval[
        df_eval["target"] != "bonafide"
    ].copy()

    # ========================================================
    # DROP IRRELEVANT ATTRIBUTES
    # ========================================================

    df.drop(
        [
            "ID",
            "target"
        ],
        axis=1,
        inplace=True
    )

    df_dev.drop(
        [
            "ID",
            "target"
        ],
        axis=1,
        inplace=True
    )

    df_eval.drop(
        [
            "ID",
            "target"
        ],
        axis=1,
        inplace=True
    )

    # ========================================================
    # PEARSON CORRELATION
    # ========================================================

    # --------------------------------------------------------
    # Training set
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 8)
    )

    corr = df.drop(
        ["spoof_attack"],
        axis=1
    ).corr(
        method="pearson"
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues"
    )

    file_path = os.path.join(
        save_dir,
        "train_pearson_corr.png"
    )

    plt.savefig(
        file_path,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Development set
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 8)
    )

    corr = df_dev.drop(
        ["spoof_attack"],
        axis=1
    ).corr(
        method="pearson"
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues"
    )

    file_path = os.path.join(
        save_dir,
        "dev_pearson_corr.png"
    )

    plt.savefig(
        file_path,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Evaluation set
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 8)
    )

    corr = df_eval.drop(
        ["spoof_attack"],
        axis=1
    ).corr(
        method="pearson"
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap="Blues"
    )

    file_path = os.path.join(
        save_dir,
        "eval_pearson_corr.png"
    )

    plt.savefig(
        file_path,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # PREPARE DATA
    # ========================================================

    data = df.values

    X = data[:, 1:]
    y = data[:, 0]

    data_dev = df_dev.values

    X_dev = data_dev[:, 1:]
    y_dev = data_dev[:, 0]

    data_eval = df_eval.values

    X_eval = data_eval[:, 1:]
    y_eval = data_eval[:, 0]

    # ========================================================
    # DECISION TREE
    # ========================================================

    clf = DecisionTreeClassifier(
        max_depth=int(
            config[
                "decision_tree"
            ][
                "max_depth"
            ]
        )
    )

    clf = clf.fit(
        X,
        y
    )

    # ========================================================
    # SAVE TRAINED MODEL
    # ========================================================

    model_path = os.path.join(
        save_dir,
        "decision_tree.pkl"
    )

    joblib.dump(
        clf,
        model_path
    )

    print(
        "\nSpoof attack decision tree saved to:"
    )

    print(
        model_path
    )

    # ========================================================
    # SAVE TREE FIGURE
    # ========================================================

    plt.figure(
        figsize=(24, 14)
    )

    plot_tree(
        clf,
        filled=True,
        feature_names=list(
            df.columns[1:]
        ),
        class_names=[
            str(c)
            for c in clf.classes_
        ],
        rounded=True
    )

    tree_path = os.path.join(
        save_dir,
        "decision_tree.png"
    )

    plt.savefig(
        tree_path,
        bbox_inches="tight"
    )

    plt.close()

    # ========================================================
    # RESULTS FILE
    # ========================================================

    text_file_path = os.path.join(
        save_dir,
        "results.txt"
    )

    with open(
        text_file_path,
        "w"
    ) as f:

        f.write(
            "SPOOF ATTACK CLASSIFICATION\n"
        )

        f.write(
            "=" * 50
            + "\n\n"
        )

    # ========================================================
    # DEVELOPMENT RESULTS
    # ========================================================

    dev_data_results(
        X_dev,
        y_dev,
        df,
        clf,
        text_file_path,
        save_dir,
        "spoof_attack"
    )

    # ========================================================
    # EVALUATION RESULTS
    # ========================================================

    eval_data_results(
        X_eval,
        y_eval,
        df,
        df_eval,
        clf,
        text_file_path,
        save_dir,
        2
    )

    print(
        "Spoof attack classification complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Decision Tree Classification "
            "for Spoofed Speech Attribution"
        )
    )

    parser.add_argument(
        "--BonafideSpoof",
        action="store_true",
        help=(
            "Run Bonafide vs Spoof "
            "classification"
        )
    )

    parser.add_argument(
        "--SpoofAttacks",
        action="store_true",
        help=(
            "Run Spoof Attack "
            "classification"
        )
    )

    args = parser.parse_args()

    # ========================================================
    # LOAD TRAINING DATA
    # ========================================================

    print(
        "\nLoading training data..."
    )

    df = pd.read_excel(
        str(
            config[
                "decision_tree"
            ][
                "train_df"
            ]
        )
    )

    print(
        "Training data shape:",
        df.shape
    )

    # ========================================================
    # LOAD DEVELOPMENT DATA
    # ========================================================

    print(
        "\nLoading development data..."
    )

    df_dev = pd.read_excel(
        str(
            config[
                "decision_tree"
            ][
                "dev_df"
            ]
        )
    )

    print(
        "Development data shape:",
        df_dev.shape
    )

    # ========================================================
    # LOAD EVALUATION DATA
    # ========================================================

    print(
        "\nLoading evaluation data..."
    )

    df_eval = pd.read_excel(
        str(
            config[
                "decision_tree"
            ][
                "eval_df"
            ]
        )
    )

    print(
        "Evaluation data shape:",
        df_eval.shape
    )

    # ========================================================
    # RESULT DIRECTORY
    # ========================================================

    save_dir = (
        "decision_tree_results"
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ========================================================
    # RUN REQUESTED TASK
    # ========================================================

    if args.BonafideSpoof:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "RUNNING BONAFIDE VS SPOOF CLASSIFICATION"
        )

        print(
            "=" * 60
        )

        bonafide_spoof(
            df,
            df_dev,
            df_eval,
            save_dir
        )

        print(
            "\nDone!"
        )

    elif args.SpoofAttacks:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "RUNNING SPOOF ATTACK CLASSIFICATION"
        )

        print(
            "=" * 60
        )

        spoof_attack_classification(
            df,
            df_dev,
            df_eval,
            save_dir
        )

        print(
            "\nDone!"
        )

    else:

        print(
            "\nNo function specified."
        )

        print(
            "Please use:"
        )

        print(
            "python decision_tree.py "
            "--BonafideSpoof"
        )

        print(
            "or:"
        )

        print(
            "python decision_tree.py "
            "--SpoofAttacks"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()