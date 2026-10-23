from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "thyroid_cleaned.csv"
OUT = BASE / "CONTROLLED_SOURCE_FREE_RESULTS"
OUT.mkdir(exist_ok=True)

TARGET = "class"
SEED_SPLIT = 2026
SEED_MODEL = 42
TEST_SIZE = 0.20
N_SPLITS = 5
N_REPEATS = 5

# Set False only if GPU is unavailable.
USE_GPU = True

# Historical EvoX-Boost hyperparameters
EVOX_PARAMS = {
    "learning_rate": 0.0343020118,
    "max_depth": 19,
    "n_estimators": 879,
    "subsample": 0.99821606,
    "colsample_bytree": 0.91857801,
    "min_child_weight": 1.0,
    "gamma": 0.49494746,
    "reg_lambda": 0.62217746,
}


# ============================================================
# METRICS
# ============================================================

def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
    }


# ============================================================
# PREPROCESSING
# ============================================================

def make_preprocessor(X_train):
    numeric_cols = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_cols = [
        c for c in X_train.columns if c not in numeric_cols
    ]

    transformers = []

    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
        ])
        transformers.append(
            ("numeric", numeric_pipe, numeric_cols)
        )

    if categorical_cols:
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=True
            )

        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ])

        transformers.append(
            ("categorical", categorical_pipe, categorical_cols)
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=True
    )


def make_model(n_classes, seed=SEED_MODEL):
    return xgb.XGBClassifier(
        **EVOX_PARAMS,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if USE_GPU else "cpu",
    )


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Place thyroid_cleaned.csv beside this script."
        )

    print(f"Loading: {DATA}", flush=True)

    df = pd.read_csv(DATA)
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    df = df.replace(
        ["?", "NA", "N/A", "na", "n/a", "", "null", "None"],
        np.nan
    )

    # IMPORTANT:
    # Do not introduce additional data-cleaning changes here.
    # This is a controlled feature-removal experiment.

    dropped = [
        c for c in ["patient_id", "dataset_name"]
        if c in df.columns
    ]
    df = df.drop(columns=dropped)

    print("Removed columns:", dropped, flush=True)
    print("Remaining raw shape:", df.shape, flush=True)

    assert "dataset_name" not in df.columns

    y_raw = df[TARGET].astype(str).str.strip()
    X = df.drop(columns=[TARGET]).copy()

    # Match the historical conversion of numeric-like columns.
    for col in X.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(X[col], errors="coerce")
        observed = X[col].notna().sum()

        if observed > 0 and converted.notna().sum() / observed >= 0.95:
            X[col] = converted

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    return X, y, y_raw, label_encoder, dropped


# ============================================================
# FIT AND EVALUATE
# ============================================================

def fit_and_predict(Xtr_raw, ytr, Xval_raw, yval,
                    n_classes, model_seed):
    # Fit preprocessing only on the current training partition.
    prep = make_preprocessor(Xtr_raw)
    Xtr = prep.fit_transform(Xtr_raw)
    Xval = prep.transform(Xval_raw)

    model = make_model(n_classes, seed=model_seed)

    start = time.time()
    model.fit(Xtr, ytr)
    elapsed = time.time() - start

    pred = np.asarray(
        model.predict(Xval)
    ).reshape(-1).astype(int)

    return model, prep, pred, elapsed


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    X, y, y_raw, label_encoder, dropped = load_data()
    classes = label_encoder.classes_
    n_classes = len(classes)

    counts = pd.Series(y).value_counts().sort_index()

    # Historical policy: classes with fewer than 5 samples
    # are retained in training but excluded from validation.
    rare_codes = counts[counts < N_SPLITS].index.to_numpy()

    rare_idx = np.flatnonzero(np.isin(y, rare_codes))
    regular_idx = np.flatnonzero(~np.isin(y, rare_codes))

    print("Classes:", list(classes), flush=True)
    print(
        "Rare classes kept training-only:",
        {
            str(classes[int(c)]): int(counts.loc[c])
            for c in rare_codes
        },
        flush=True
    )

    # Independent holdout split.
    train_regular, test_idx = train_test_split(
        regular_idx,
        test_size=TEST_SIZE,
        random_state=SEED_SPLIT,
        stratify=y[regular_idx]
    )

    train_idx = np.concatenate([train_regular, rare_idx])

    X_train = X.iloc[train_idx].copy()
    y_train = y[train_idx]
    X_test = X.iloc[test_idx].copy()
    y_test = y[test_idx]

    pd.DataFrame({
        "row_index": train_idx,
        "partition": "train",
        "label": y_raw.iloc[train_idx].to_numpy()
    }).to_csv(OUT / "train_indices.csv", index=False)

    pd.DataFrame({
        "row_index": test_idx,
        "partition": "holdout",
        "label": y_raw.iloc[test_idx].to_numpy()
    }).to_csv(OUT / "holdout_indices.csv", index=False)

    print(
        f"Training records: {len(train_idx)}; "
        f"holdout records: {len(test_idx)}",
        flush=True
    )

    # --------------------------------------------------------
    # HOLDOUT EVALUATION
    # --------------------------------------------------------

    print("\nTraining controlled source-free EvoX-Boost...", flush=True)

    model, prep, y_pred, elapsed = fit_and_predict(
        X_train, y_train,
        X_test, y_test,
        n_classes, SEED_MODEL
    )

    holdout_metrics = evaluate(y_test, y_pred)
    holdout_metrics["fit_seconds"] = elapsed

    print("\nHOLDOUT METRICS", flush=True)
    print(json.dumps(holdout_metrics, indent=2), flush=True)

    with open(OUT / "holdout_metrics.json", "w") as f:
        json.dump(holdout_metrics, f, indent=2)

    predictions = pd.DataFrame({
        "row_index": test_idx,
        "true_code": y_test,
        "true_label": label_encoder.inverse_transform(y_test),
        "predicted_code": y_pred,
        "predicted_label": label_encoder.inverse_transform(y_pred),
    })
    predictions.to_csv(
        OUT / "holdout_predictions.csv", index=False
    )

    cm = confusion_matrix(
        y_test, y_pred, labels=np.arange(n_classes)
    )
    pd.DataFrame(
        cm, index=classes, columns=classes
    ).to_csv(OUT / "holdout_confusion_matrix.csv")

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(n_classes),
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        OUT / "holdout_per_class_report.csv"
    )

    # Save native XGBoost contributions for later SHAP analysis.
    print("\nExporting native XGBoost contributions...", flush=True)

    import xgboost as xgb

    X_test_processed = prep.transform(X_test)
    dtest = xgb.DMatrix(X_test_processed)

    contributions = model.get_booster().predict(
        dtest, pred_contribs=True
    )
    np.save(
        OUT / "native_contributions.npy",
        contributions
    )

    pd.Series(
        prep.get_feature_names_out(),
        name="feature_name"
    ).to_csv(OUT / "feature_names.csv", index=False)

    pd.DataFrame({
        "class_code": np.arange(n_classes),
        "class_label": classes
    }).to_csv(OUT / "class_mapping.csv", index=False)

    print(
        "Contribution shape:",
        contributions.shape,
        flush=True
    )

    # --------------------------------------------------------
    # REPEATED STRATIFIED CROSS-VALIDATION
    # --------------------------------------------------------

    print(
        f"\nStarting {N_SPLITS}-fold x {N_REPEATS}-repeat CV...",
        flush=True
    )

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED_SPLIT
    )

    X_regular = X.iloc[regular_idx].reset_index(drop=True)
    y_regular = y[regular_idx]

    rows = []
    total = N_SPLITS * N_REPEATS

    for fold, (tr_rel, val_rel) in enumerate(
        cv.split(X_regular, y_regular), start=1
    ):
        tr_idx = np.concatenate([
            regular_idx[tr_rel], rare_idx
        ])
        val_idx = regular_idx[val_rel]

        _, _, pred, fit_seconds = fit_and_predict(
            X.iloc[tr_idx].copy(),
            y[tr_idx],
            X.iloc[val_idx].copy(),
            y[val_idx],
            n_classes,
            SEED_MODEL
        )

        result = evaluate(y[val_idx], pred)
        result.update({
            "fold": fold,
            "repeat": (fold - 1) // N_SPLITS + 1,
            "fit_seconds": fit_seconds,
            "train_n": len(tr_idx),
            "validation_n": len(val_idx)
        })
        rows.append(result)

        print(
            f"Fold {fold}/{total}: "
            f"accuracy={result['accuracy']:.5f}, "
            f"macro-F1={result['macro_f1']:.5f}, "
            f"balanced accuracy={result['balanced_accuracy']:.5f}",
            flush=True
        )

        pd.DataFrame(rows).to_csv(
            OUT / "cv_fold_metrics.csv", index=False
        )

    cv_df = pd.DataFrame(rows)

    cv_df.to_csv(OUT / "cv_fold_metrics.csv", index=False)

    metric_cols = [
        "accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "balanced_accuracy"
    ]

    summary = cv_df[metric_cols].agg(["mean", "std"]).T
    summary.to_csv(OUT / "cv_summary.csv")

    repeat_means = cv_df.groupby("repeat")[metric_cols].mean()
    repeat_means.to_csv(OUT / "cv_repeat_means.csv")

    print("\nCV SUMMARY: mean and SD across folds", flush=True)
    print(summary.to_string(), flush=True)

    print(
        f"\nCOMPLETE. Results saved in: {OUT}",
        flush=True
    )


if __name__ == "__main__":
    main()