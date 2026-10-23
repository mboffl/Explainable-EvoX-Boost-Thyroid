
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION: MATCH HISTORICAL 3Z-E2 PIPELINE
# ============================================================

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "thyroid_cleaned.csv"
OUTPUT_DIR = BASE / "PAIRED_SOURCE_ABLATION"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "class"
TEST_SIZE = 0.20
SPLIT_SEED = 2026
MODEL_SEED = 42
N_SPLITS = 5

# Original published EvoX-Boost parameters
PARAMS = {
    "learning_rate": 0.0343020118,
    "max_depth": 19,
    "subsample": 0.99821606,
    "colsample_bytree": 0.91857801,
    "n_estimators": 879,
    "min_child_weight": 1.0,
    "gamma": 0.49494746,
    "reg_lambda": 0.62217746,
}

USE_GPU = True


# ============================================================
# HISTORICAL PREPROCESSOR
# ============================================================

def make_preprocessor(X_train):
    numeric_features = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ])

        transformers.append(
            ("num", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        try:
            encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
            )
        except TypeError:
            encoder = OneHotEncoder(
                sparse=False,
                handle_unknown="ignore",
            )

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ])

        transformers.append(
            ("cat", categorical_pipeline, categorical_features)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# ============================================================
# HISTORICAL MODEL
# ============================================================

def make_model(n_classes):
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        learning_rate=PARAMS["learning_rate"],
        max_depth=PARAMS["max_depth"],
        subsample=PARAMS["subsample"],
        colsample_bytree=PARAMS["colsample_bytree"],
        n_estimators=PARAMS["n_estimators"],
        min_child_weight=PARAMS["min_child_weight"],
        gamma=PARAMS["gamma"],
        reg_lambda=PARAMS["reg_lambda"],
        tree_method="hist",
        device="cuda" if USE_GPU else "cpu",
        random_state=MODEL_SEED,
        eval_metric="mlogloss",
        n_jobs=-1,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true, y_pred,
            average="weighted",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
    }


# ============================================================
# RUN ONE FEATURE CONDITION
# ============================================================

def run_condition(condition, X, y, train_idx, test_idx, classes):
    print("\n" + "=" * 72, flush=True)
    print(f"RUNNING: {condition}", flush=True)
    print("=" * 72, flush=True)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Raw training shape:", X_train.shape, flush=True)
    print("Raw holdout shape:", X_test.shape, flush=True)

    # Fit imputation, scaling and encoding on training data only.
    preprocessor = make_preprocessor(X_train)

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    print(
        "Processed shapes:",
        X_train_proc.shape,
        X_test_proc.shape,
        flush=True,
    )

    # Historical handling for any class absent from training.
    present_classes = np.unique(y_train)
    missing_classes = sorted(
        set(range(len(classes))) - set(present_classes)
    )

    y_model_train = y_train.copy()
    X_model_train = X_train_proc

    if missing_classes:
        print(
            "Missing training classes; adding historical dummy rows:",
            [classes[c] for c in missing_classes],
            flush=True,
        )

        dummy_X = np.zeros(
            (len(missing_classes), X_train_proc.shape[1]),
            dtype=X_train_proc.dtype,
        )
        dummy_y = np.asarray(missing_classes, dtype=int)

        X_model_train = np.vstack([X_train_proc, dummy_X])
        y_model_train = np.concatenate([y_train, dummy_y])

    model = make_model(len(classes))

    start = time.time()
    model.fit(X_model_train, y_model_train)
    fit_seconds = time.time() - start

    y_pred = np.asarray(
        model.predict(X_test_proc)
    ).reshape(-1).astype(int)

    metrics = calculate_metrics(y_test, y_pred)
    metrics.update({
        "condition": condition,
        "processed_feature_count": int(X_train_proc.shape[1]),
        "fit_seconds": fit_seconds,
        "holdout_size": int(len(y_test)),
        "predicted_class_count": int(np.unique(y_pred).size),
    })

    print("\nMetrics:", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)

    print("\nPredicted class distribution:", flush=True)
    print(
        pd.Series(
            classes[y_pred]
        ).value_counts().to_string(),
        flush=True,
    )

    print("\nClassification report:", flush=True)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    print(report_df.to_string(), flush=True)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(classes)),
    )

    pd.DataFrame(
        cm, index=classes, columns=classes
    ).to_csv(
        OUTPUT_DIR / f"{condition}_confusion_matrix.csv"
    )

    report_df.to_csv(
        OUTPUT_DIR / f"{condition}_classification_report.csv"
    )

    predictions = pd.DataFrame({
        "row_index": test_idx,
        "true_code": y_test,
        "true_label": classes[y_test],
        "predicted_code": y_pred,
        "predicted_label": classes[y_pred],
    })

    predictions.to_csv(
        OUTPUT_DIR / f"{condition}_predictions.csv",
        index=False,
    )

    pd.Series(
        preprocessor.get_feature_names_out(),
        name="feature_name",
    ).to_csv(
        OUTPUT_DIR / f"{condition}_feature_names.csv",
        index=False,
    )

    return metrics, model, preprocessor


# ============================================================
# MAIN: IDENTICAL ROWS, ONLY dataset_name CHANGES
# ============================================================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print("Loading:", DATA_PATH, flush=True)

    df = pd.read_csv(DATA_PATH)

    # Match original script: no extra missing-token replacement
    # or age correction is introduced here.
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET!r} not found.")

    if "dataset_name" not in df.columns:
        raise ValueError(
            "dataset_name is absent from the raw dataset; "
            "the paired experiment cannot be performed."
        )

    y_raw = df[TARGET].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    classes = label_encoder.classes_
    n_classes = len(classes)

    # Original feature condition removes patient_id only.
    X_original = df.drop(
        columns=[TARGET, "patient_id"],
        errors="ignore",
    ).copy()

    # Source-free condition differs only by dataset_name removal.
    X_source_free = X_original.drop(
        columns=["dataset_name"]
    ).copy()

    assert "dataset_name" in X_original.columns
    assert "dataset_name" not in X_source_free.columns

    print("Dataset rows:", len(df), flush=True)
    print("Class count:", n_classes, flush=True)
    print("Original features:", X_original.shape[1], flush=True)
    print("Source-free features:", X_source_free.shape[1], flush=True)

    # Preserve the historical rare-class rule.
    counts = pd.Series(y).value_counts().sort_index()
    rare_codes = counts[counts < N_SPLITS].index.to_numpy()

    rare_mask = np.isin(y, rare_codes)
    regular_idx = np.flatnonzero(~rare_mask)
    rare_idx = np.flatnonzero(rare_mask)

    print(
        "Rare classes kept training-only:",
        {
            str(classes[int(c)]): int(counts.loc[c])
            for c in rare_codes
        },
        flush=True,
    )

    # Make the split once and reuse exact row indices for both models.
    train_regular, test_idx = train_test_split(
        regular_idx,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=y[regular_idx],
    )

    train_idx = np.concatenate([train_regular, rare_idx])

    assert len(test_idx) == 4526, (
        f"Unexpected holdout size: {len(test_idx)}"
    )

    pd.DataFrame({
        "row_index": train_idx,
        "partition": "train",
        "target": y_raw.iloc[train_idx].to_numpy(),
    }).to_csv(OUTPUT_DIR / "shared_train_indices.csv", index=False)

    pd.DataFrame({
        "row_index": test_idx,
        "partition": "holdout",
        "target": y_raw.iloc[test_idx].to_numpy(),
    }).to_csv(OUTPUT_DIR / "shared_holdout_indices.csv", index=False)

    # Run both conditions with the exact same row partitions.
    original_metrics, _, _ = run_condition(
        "with_dataset_name",
        X_original, y,
        train_idx, test_idx, classes,
    )

    source_free_metrics, _, _ = run_condition(
        "without_dataset_name",
        X_source_free, y,
        train_idx, test_idx, classes,
    )

    comparison = pd.DataFrame([
        original_metrics,
        source_free_metrics,
    ])

    comparison.to_csv(
        OUTPUT_DIR / "paired_ablation_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 72, flush=True)
    print("PAIRED ABLATION COMPARISON", flush=True)
    print("=" * 72, flush=True)
    print(comparison.to_string(index=False), flush=True)

    with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
        json.dump({
            "dataset": DATA_PATH.name,
            "rows": int(len(df)),
            "target": TARGET,
            "classes": [str(c) for c in classes],
            "removed_patient_id": True,
            "condition_A_retains_dataset_name": True,
            "condition_B_removes_dataset_name": True,
            "same_train_indices": True,
            "same_holdout_indices": True,
            "split_seed": SPLIT_SEED,
            "model_seed": MODEL_SEED,
            "parameters": PARAMS,
            "gpu_requested": USE_GPU,
            "note": (
                "Paired holdout ablation. Both conditions use "
                "the same split and preprocessing protocol."
            ),
        }, f, indent=2)

    print(
        f"\nCOMPLETE. Outputs saved to: {OUTPUT_DIR}",
        flush=True,
    )


if __name__ == "__main__":
    main()
