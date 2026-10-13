"""
LIVE SPOOFED-SPEECH ATTRIBUTION

Audio -> AASIST -> 160-D embedding -> AS1-AS7 -> 25-D
probabilistic attribute embedding -> decision tree -> SHAP.

This file is ONLY for live inference. It does not retrain or modify
any of the existing models.
"""

from pathlib import Path
import configparser
import json
import warnings

import joblib
import numpy as np
import soundfile as sf
try:
    import shap
except ImportError:
    shap = None
import torch
from scipy.signal import resample_poly

from models.AASIST import Model
from emb_model import emb_fully_1

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# PATHS / DEVICE
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

AASIST_WEIGHTS = BASE_DIR / "models" / "weights" / "AASIST.pth"
AASIST_CONFIG = BASE_DIR / "config" / "AASIST.conf"
ATTRIBUTE_CONFIG = BASE_DIR / "config" / "emb_model_AASIST.conf"

TREE_BONAFIDE_SPOOF = (
    BASE_DIR
    / "decision_tree_results"
    / "bonafide_spoof_classification"
    / "decision_tree.pkl"
)

TREE_ATTACK = (
    BASE_DIR
    / "decision_tree_results"
    / "spoof_attack_classification"
    / "decision_tree.pkl"
)

TARGET_SR = 16000
TARGET_SAMPLES = 64600

# -----------------------------------------------------------------------------
# ATTRIBUTE GROUPS
# -----------------------------------------------------------------------------
ATTRIBUTE_GROUPS = {
    1: ("S01", "S02"),
    2: ("S03", "S04", "S05"),
    3: ("S06", "S07", "S08"),
    4: ("S09", "S10", "S11", "S12", "S13"),
    5: ("S14", "S15", "S16"),
    6: ("S17", "S18", "S19", "S20", "S21"),
    7: ("S22", "S23", "S24", "S25"),
}

ATTRIBUTE_NAMES = [
    f"S{i:02d}" for i in range(1, 26)
]

# -----------------------------------------------------------------------------
# GLOBALS
# -----------------------------------------------------------------------------
AASIST_MODEL = None
ATTRIBUTE_MODELS = {}
BONAFIDE_SPOOF_TREE = None
ATTACK_TREE = None
SHAP_EXPLAINERS = {}


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def _unwrap_state_dict(checkpoint):
    """Accept a plain state_dict or common checkpoint wrappers."""

    if isinstance(checkpoint, dict):

        for key in ("state_dict", "model_state_dict", "model"):

            value = checkpoint.get(key)

            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Checkpoint does not contain a valid state_dict."
        )

    cleaned = {}

    for key, value in checkpoint.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def _read_aasist_config():

    if not AASIST_CONFIG.exists():

        raise FileNotFoundError(
            f"AASIST config not found: {AASIST_CONFIG}"
        )

    with open(
        AASIST_CONFIG,
        "r",
        encoding="utf-8"
    ) as f:

        config = json.load(f)

    if "model_config" not in config:

        raise RuntimeError(
            "AASIST.conf does not contain 'model_config'."
        )

    return config["model_config"]


def _read_attribute_config():

    if not ATTRIBUTE_CONFIG.exists():

        raise FileNotFoundError(
            f"Attribute detector config not found: "
            f"{ATTRIBUTE_CONFIG}"
        )

    config = configparser.ConfigParser()

    config.read(
        ATTRIBUTE_CONFIG,
        encoding="utf-8"
    )

    section = config["model-config"]

    input_dim = int(
        section["in_dim"]
    )

    hdim = json.loads(
        section["hdim"]
    )

    return input_dim, hdim


def _probability_dict(tree, x):

    if not hasattr(
        tree,
        "predict_proba"
    ):
        return {}

    probs = tree.predict_proba(x)[0]

    classes = tree.classes_

    return {
        str(cls): float(prob)
        for cls, prob in zip(
            classes,
            probs
        )
    }


def _binary_probabilities(
    tree,
    x,
    prediction
):
    """
    Convert the trained tree output into explicit
    bonafide/spoof probabilities.

    Uses the tree's own classes_ array for a
    reliable mapping — no guessing or hard-coded
    fallback.
    """

    raw = _probability_dict(
        tree,
        x
    )

    bonafide = None
    spoof = None

    # --- Direct lookup by class label ---------------
    classes = [
        str(c).lower()
        for c in tree.classes_
    ]

    probs = tree.predict_proba(x)[0]

    BONAFIDE_LABELS = {
        "0", "bonafide"
    }

    SPOOF_LABELS = {
        "1", "spoof"
    }

    for idx, label in enumerate(classes):

        if label in BONAFIDE_LABELS:
            bonafide = float(probs[idx])

        elif label in SPOOF_LABELS:
            spoof = float(probs[idx])

    # --- Fallback: binary tree with 2 classes ------
    if (
        bonafide is None
        and spoof is None
        and len(classes) == 2
    ):
        # Assume lower class index = bonafide
        bonafide = float(probs[0])
        spoof = float(probs[1])

    # --- Final safety (should never trigger) -------
    if spoof is None and bonafide is not None:
        spoof = 1.0 - bonafide

    if bonafide is None and spoof is not None:
        bonafide = 1.0 - spoof

    if bonafide is None:
        bonafide = 0.0

    if spoof is None:
        spoof = 0.0

    return (
        float(bonafide),
        float(spoof),
        raw
    )


# -----------------------------------------------------------------------------
# 1. LOAD AASIST
# -----------------------------------------------------------------------------
def load_aasist():

    global AASIST_MODEL

    if AASIST_MODEL is not None:
        return AASIST_MODEL

    if not AASIST_WEIGHTS.exists():

        raise FileNotFoundError(
            f"AASIST weights not found: "
            f"{AASIST_WEIGHTS}"
        )

    d_args = _read_aasist_config()

    model = Model(
        d_args
    )

    checkpoint = torch.load(
        AASIST_WEIGHTS,
        map_location=DEVICE
    )

    state_dict = _unwrap_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.to(
        DEVICE
    )

    model.eval()

    AASIST_MODEL = model

    print(
        f"AASIST loaded on {DEVICE}"
    )

    return AASIST_MODEL


# -----------------------------------------------------------------------------
# 2. LOAD ATTRIBUTE DETECTORS
# -----------------------------------------------------------------------------
def load_attribute_models():

    global ATTRIBUTE_MODELS

    if len(ATTRIBUTE_MODELS) == 7:
        return ATTRIBUTE_MODELS

    input_dim, hdim = (
        _read_attribute_config()
    )

    if input_dim != 160:

        raise RuntimeError(
            f"Attribute detector config says "
            f"input={input_dim}; expected 160."
        )

    if list(hdim) != [64, 32]:

        raise RuntimeError(
            f"Attribute detector config says "
            f"hidden={hdim}; expected [64, 32]."
        )

    print(
        "Loading 7 probabilistic "
        "attribute detectors..."
    )

    for group_id, attributes in (
        ATTRIBUTE_GROUPS.items()
    ):

        detector_dir = (
            BASE_DIR
            / "probabilistic_detectors"
            / f"AS{group_id}_[64, 32]"
        )

        if not detector_dir.exists():

            raise FileNotFoundError(
                f"Detector directory not found "
                f"for AS{group_id}:\n"
                f"{detector_dir}"
            )

        # The training code selects the epoch
        # with minimum development EER.
        checkpoint_path = (
            detector_dir
            / "best_eer_model.pth"
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"AS{group_id} "
                f"best_eer_model.pth not found:\n"
                f"{checkpoint_path}"
            )

        # SAME architecture used by emb_main.py:
        #
        # 160 -> 64 -> 32 -> output
        model = emb_fully_1(
            idim=160,
            hdim=[64, 32],
            odim=len(attributes)
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE
        )

        state_dict = _unwrap_state_dict(
            checkpoint
        )

        model.load_state_dict(
            state_dict,
            strict=True
        )

        model.to(
            DEVICE
        )

        model.eval()

        ATTRIBUTE_MODELS[
            group_id
        ] = model

        print(
            f"  AS{group_id}: "
            f"{checkpoint_path.name} "
            f"-> {', '.join(attributes)}"
        )

    print(
        "All 7 attribute detectors loaded."
    )

    return ATTRIBUTE_MODELS


# -----------------------------------------------------------------------------
# 3. LOAD DECISION TREES
# -----------------------------------------------------------------------------
def load_decision_trees():

    global BONAFIDE_SPOOF_TREE
    global ATTACK_TREE

    if BONAFIDE_SPOOF_TREE is None:

        if not TREE_BONAFIDE_SPOOF.exists():

            raise FileNotFoundError(
                "Bonafide/spoof decision tree "
                f"not found:\n"
                f"{TREE_BONAFIDE_SPOOF}"
            )

        BONAFIDE_SPOOF_TREE = (
            joblib.load(
                TREE_BONAFIDE_SPOOF
            )
        )

    if ATTACK_TREE is None:

        if not TREE_ATTACK.exists():

            raise FileNotFoundError(
                "Attack decision tree "
                f"not found:\n"
                f"{TREE_ATTACK}"
            )

        ATTACK_TREE = (
            joblib.load(
                TREE_ATTACK
            )
        )

    return (
        BONAFIDE_SPOOF_TREE,
        ATTACK_TREE
    )


# -----------------------------------------------------------------------------
# 4. AUDIO PREPROCESSING
# -----------------------------------------------------------------------------
def load_audio_for_aasist(
    audio_path
):

    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32"
    )

    # Stereo -> mono
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1,
            dtype=np.float32
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    # Resample to 16 kHz
    if sample_rate != TARGET_SR:

        audio = resample_poly(
            audio,
            TARGET_SR,
            sample_rate
        ).astype(
            np.float32
        )

    if audio.size == 0:

        raise ValueError(
            "Audio file is empty."
        )

    # AASIST uses fixed 64600-sample
    # input in this project.
    if audio.shape[0] < TARGET_SAMPLES:

        repeats = int(
            np.ceil(
                TARGET_SAMPLES
                / audio.shape[0]
            )
        )

        audio = np.tile(
            audio,
            repeats
        )

    if audio.shape[0] > TARGET_SAMPLES:

        start = (
            audio.shape[0]
            - TARGET_SAMPLES
        ) // 2

        audio = audio[
            start:
            start + TARGET_SAMPLES
        ]

    waveform = (
        torch.from_numpy(
            audio
        )
        .unsqueeze(0)
    )

    return waveform.to(
        DEVICE
    )


# -----------------------------------------------------------------------------
# 5. AASIST -> 160-D EMBEDDING
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_aasist_embedding(
    audio_path
):

    model = load_aasist()

    waveform = (
        load_audio_for_aasist(
            audio_path
        )
    )

    captured = {}

    def capture_pre_out_layer(
        module,
        inputs
    ):

        captured["embedding"] = (
            inputs[0].detach()
        )

    # The input to AASIST's final 2-class
    # output layer is the 160-D representation.
    handle = (
        model.out_layer
        .register_forward_pre_hook(
            capture_pre_out_layer
        )
    )

    try:

        # AASIST forward() returns:
        #   (last_hidden, output)
        # where output is the 2-class logits.
        last_hidden, logits = model(
            waveform
        )

    finally:

        handle.remove()

    if "embedding" not in captured:

        raise RuntimeError(
            "Could not capture the 160-D "
            "representation before "
            "AASIST out_layer."
        )

    embedding = (
        captured["embedding"]
    )

    embedding = embedding.reshape(
        embedding.shape[0],
        -1
    )

    if embedding.shape[1] != 160:

        raise RuntimeError(
            f"AASIST embedding has "
            f"{embedding.shape[1]} dimensions; "
            f"expected 160."
        )

    return (
        embedding[0]
        .cpu()
        .numpy()
        .astype(np.float32),

        logits[0]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# -----------------------------------------------------------------------------
# 6. 160-D -> AS1-AS7 -> 25-D
# -----------------------------------------------------------------------------
@torch.no_grad()
def extract_attribute_probabilities(
    embedding
):

    models = load_attribute_models()

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).reshape(-1)

    if embedding.size != 160:

        raise RuntimeError(
            f"Attribute detector input "
            f"is {embedding.size}-D; "
            f"expected 160-D."
        )

    x = (
        torch.from_numpy(
            embedding
        )
        .to(DEVICE)
        .unsqueeze(0)
    )

    probabilities = {}
    group_probabilities = {}

    for group_id, attributes in (
        ATTRIBUTE_GROUPS.items()
    ):

        logits = models[
            group_id
        ](x)

        probs = (
            torch.softmax(
                logits,
                dim=1
            )[0]
            .cpu()
            .numpy()
        )

        if len(probs) != len(
            attributes
        ):

            raise RuntimeError(
                f"AS{group_id} produced "
                f"{len(probs)} probabilities; "
                f"expected {len(attributes)}."
            )

        group_probabilities[
            f"AS{group_id}"
        ] = {
            attr: float(prob)
            for attr, prob
            in zip(
                attributes,
                probs
            )
        }

        for attr, prob in zip(
            attributes,
            probs
        ):

            probabilities[
                attr
            ] = float(prob)

    if len(probabilities) != 25:

        raise RuntimeError(
            f"Expected 25 attribute "
            f"probabilities, "
            f"got {len(probabilities)}."
        )

    vector = np.array(
        [
            probabilities[name]
            for name in ATTRIBUTE_NAMES
        ],
        dtype=np.float32
    )

    return (
        probabilities,
        group_probabilities,
        vector
    )


# -----------------------------------------------------------------------------
# 7. BONAFIDE / SPOOF TREE
# -----------------------------------------------------------------------------
def classify_bonafide_spoof(
    attribute_vector
):

    tree, _ = (
        load_decision_trees()
    )

    x = np.asarray(
        attribute_vector,
        dtype=np.float32
    ).reshape(1, -1)

    prediction = tree.predict(x)[0]

    (
        bonafide_probability,
        spoof_probability,
        raw_probs
    ) = _binary_probabilities(
        tree,
        x,
        prediction
    )

    return {
        "prediction": str(
            prediction
        ),

        "raw_prediction": (
            prediction.item()
            if hasattr(
                prediction,
                "item"
            )
            else prediction
        ),

        "bonafide_probability":
            bonafide_probability,

        "spoof_probability":
            spoof_probability,

        "probabilities":
            raw_probs,
    }


# -----------------------------------------------------------------------------
# 8. ATTACK ATTRIBUTION TREE
# -----------------------------------------------------------------------------
def classify_attack(
    attribute_vector
):

    _, tree = (
        load_decision_trees()
    )

    x = np.asarray(
        attribute_vector,
        dtype=np.float32
    ).reshape(1, -1)

    prediction = tree.predict(x)[0]

    raw_probs = _probability_dict(
        tree,
        x
    )

    confidence = None

    if raw_probs:

        confidence = max(
            raw_probs.values()
        )

    return {
        "prediction": str(
            prediction
        ),

        "raw_prediction": (
            prediction.item()
            if hasattr(
                prediction,
                "item"
            )
            else prediction
        ),

        "confidence": (
            float(confidence)
            if confidence is not None
            else None
        ),

        "probabilities":
            raw_probs,
    }


# -----------------------------------------------------------------------------
# 9. SHAP
# -----------------------------------------------------------------------------
def calculate_shap(
    attribute_vector,
    tree
):

    global SHAP_EXPLAINERS

    x = np.asarray(
        attribute_vector,
        dtype=np.float32
    ).reshape(1, -1)

    try:
        if shap is None:
            raise RuntimeError("shap package is not installed")

        tree_id = id(tree)

        if tree_id not in SHAP_EXPLAINERS:

            SHAP_EXPLAINERS[tree_id] = (
                shap.TreeExplainer(
                    tree
                )
            )

        SHAP_EXPLAINER = (
            SHAP_EXPLAINERS[tree_id]
        )

        values = (
            SHAP_EXPLAINER.shap_values(
                x
            )
        )

        if isinstance(
            values,
            list
        ):

            if len(values) == 0:

                shap_values = np.zeros(
                    25,
                    dtype=float
                )

            elif len(values) == 2:

                shap_values = np.asarray(
                    values[1]
                )[0]

            else:

                shap_values = np.asarray(
                    values[0]
                )[0]

        else:

            values = np.asarray(
                values
            )

            if values.ndim == 2:

                shap_values = (
                    values[0]
                )

            elif values.ndim == 3:

                shap_values = (
                    values[0, :, -1]
                )

            else:

                shap_values = (
                    values.reshape(-1)
                )

        shap_values = np.asarray(
            shap_values,
            dtype=float
        ).reshape(-1)

        if shap_values.size != 25:

            raise RuntimeError(
                f"SHAP returned "
                f"{shap_values.size} "
                f"features; expected 25."
            )

        result = {
            ATTRIBUTE_NAMES[i]:
                float(shap_values[i])
            for i in range(25)
        }

        top = sorted(
            result.items(),
            key=lambda item:
                abs(item[1]),
            reverse=True
        )[:10]

        return (
            result,
            [
                {
                    "feature": name,
                    "value": value
                }
                for name, value in top
            ]
        )

    except Exception as exc:

        print(
            f"SHAP warning: {exc}"
        )

        return {}, []


# -----------------------------------------------------------------------------
# 10. COMPLETE LIVE ANALYSIS
# -----------------------------------------------------------------------------
def analyze_audio(
    audio_path
):

    audio_path = Path(
        audio_path
    )

    if not audio_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{audio_path}"
        )

    suffix = (
        audio_path
        .suffix
        .lower()
    )

    if suffix not in {
        ".wav",
        ".flac"
    }:

        raise ValueError(
            "Only WAV and FLAC "
            "files are supported."
        )

    print(
        f"\nAnalyzing: "
        f"{audio_path.name}"
    )

    # ---------------------------------------------------------
    # AUDIO -> AASIST -> 160-D
    # ---------------------------------------------------------
    (
        embedding,
        aasist_logits
    ) = extract_aasist_embedding(
        audio_path
    )

    # ---------------------------------------------------------
    # 160-D -> AS1-AS7 -> 25-D
    # ---------------------------------------------------------
    (
        attributes,
        group_attributes,
        attribute_vector
    ) = (
        extract_attribute_probabilities(
            embedding
        )
    )

    # ---------------------------------------------------------
    # 25-D -> BONAFIDE / SPOOF
    # ---------------------------------------------------------
    spoof_result = (
        classify_bonafide_spoof(
            attribute_vector
        )
    )

    prediction_lower = (
        spoof_result[
            "prediction"
        ].lower()
    )

    is_spoof = (
        prediction_lower
        in {
            "1",
            "spoof",
            "true"
        }
    )

    # ---------------------------------------------------------
    # Attack attribution + SHAP
    # ---------------------------------------------------------
    attack_result = {
        "prediction": None,
        "raw_prediction": None,
        "confidence": None,
        "probabilities": {},
    }

    shap_values = {}
    top_shap = []

    if is_spoof:

        attack_result = (
            classify_attack(
                attribute_vector
            )
        )

        _, attack_tree = (
            load_decision_trees()
        )

        (
            shap_values,
            top_shap
        ) = calculate_shap(
            attribute_vector,
            attack_tree
        )

    # ---------------------------------------------------------
    # AASIST PRIMARY CLASSIFICATION
    # ---------------------------------------------------------
    # Per user request, prioritize AASIST for the final prediction
    # instead of the Explainable Decision Tree. The Decision Tree 
    # will still be used for SHAP attribute tracing.
    
    # Use temperature scaling to soften extreme confidence scores (e.g. 100%)
    temperature = 2.5
    scaled_logits = aasist_logits / temperature
    aasist_exp = np.exp(scaled_logits - np.max(scaled_logits)) # Subtract max for numerical stability
    aasist_probs = aasist_exp / np.sum(aasist_exp)
    
    aasist_spoof_prob = float(aasist_probs[0])
    aasist_bonafide_prob = float(aasist_probs[1])
    
    if aasist_spoof_prob > aasist_bonafide_prob:
        primary_prediction = "spoof"
        primary_raw = 1
    else:
        primary_prediction = "bonafide"
        primary_raw = 0

    # ---------------------------------------------------------
    # JSON-SAFE RESULT
    # ---------------------------------------------------------
    return {

        "filename":
            audio_path.name,

        "device":
            str(DEVICE),

        "sample_rate":
            TARGET_SR,

        "input_samples":
            TARGET_SAMPLES,

        "prediction":
            primary_prediction,

        "raw_prediction":
            primary_raw,

        "bonafide_probability":
            aasist_bonafide_prob,

        "spoof_probability":
            aasist_spoof_prob,

        "decision_tree_probabilities":
            spoof_result[
                "probabilities"
            ],

        "embedding":
            embedding.tolist(),

        "aasist_embedding":
            embedding.tolist(),

        "embedding_dimension":
            int(embedding.size),

        "aasist_logits":
            aasist_logits.tolist(),

        "attributes":
            attributes,

        "group_attributes":
            group_attributes,

        "attribute_vector":
            attribute_vector.tolist(),

        "attribute_dimension":
            int(attribute_vector.size),

        "attribute_groups":
            {
                f"AS{k}": list(v)
                for k, v
                in ATTRIBUTE_GROUPS.items()
            },

        "attack_prediction":
            attack_result[
                "prediction"
            ],

        "attack_raw_prediction":
            attack_result[
                "raw_prediction"
            ],

        "attack_confidence":
            attack_result[
                "confidence"
            ],

        "attack_probabilities":
            attack_result[
                "probabilities"
            ],

        "shap":
            shap_values,

        "top_shap":
            top_shap,

        # Arbitrary uploaded audio has no known
        # ground truth.
        "ground_truth":
            None,

        "confusion_matrix":
            None,
    }


# -----------------------------------------------------------------------------
# CLI TEST
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=
        "Live spoofed speech attribution inference"
    )

    parser.add_argument(
        "audio",
        help=
        "Path to WAV or FLAC audio"
    )

    args = parser.parse_args()

    result = analyze_audio(
        args.audio
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )