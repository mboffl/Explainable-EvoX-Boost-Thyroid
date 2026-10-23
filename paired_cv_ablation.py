
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "thyroid_cleaned.csv"
OUT = BASE / "PAIRED_CV_ABLATION"
OUT.mkdir(exist_ok=True)

TARGET = "class"
SEED = 2026
MODEL_SEED = 42
N_SPLITS = 5
N_REPEATS = 5
USE_GPU = True

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


def make_preprocessor(X):
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    transformers = []

    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="mean")),
            ]),
            numeric
        ))

    if categorical:
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

        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", encoder),
            ]),
            categorical
        ))

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=True
    )


def make_model(n_classes, seed=MODEL_SEED):
    return xgb.XGBClassifier(
        **PARAMS,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if USE_GPU else "cpu"
    )


def get_metrics(y_true, y_pred):
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


def fit_condition(X, y, train_idx, val_idx, condition, fold):
    Xtr_raw = X.iloc[train_idx].copy()
    Xval_raw = X.iloc[val_idx].copy()
    ytr = y[train_idx]
    yval = y[val_idx]

    prep = make_preprocessor(Xtr_raw)
    Xtr = prep.fit_transform(Xtr_raw)
    Xval = prep.transform(Xval_raw)

    model = make_model(len(np.unique(y)), MODEL_SEED)

    start = time.time()
    model.fit(Xtr, ytr)
    elapsed = time.time() - start

    pred = np.asarray(model.predict(Xval)).reshape(-1).astype(int)

    result = get_metrics(yval, pred)
    result.update({
        "condition": condition,
        "fold": fold,
        "repeat": (fold - 1) // N_SPLITS + 1,
        "train_n": len(train_idx),
        "validation_n": len(val_idx),
        "fit_seconds": elapsed,
        "predicted_class_count": int(np.unique(pred).size),
    })

    print(
        f"{condition} | fold {fold}/{N_SPLITS*N_REPEATS} | "
        f"accuracy={result['accuracy']:.5f} | "
        f"macro-F1={result['macro_f1']:.5f} | "
        f"balanced-accuracy={result['balanced_accuracy']:.5f}",
        flush=True
    )

    # Return feature importance on each fitted fold.
    booster = model.get_booster()
    importance = booster.get_score(importance_type="gain")
    importance_rows = []

    names = prep.get_feature_names_out()
    for key, value in importance.items():
        # XGBoost may report feature indices as f0, f1, ...
        try:
            index = int(key[1:])
            feature_name = names[index]
        except (ValueError, IndexError):
            feature_name = key

        importance_rows.append({
            "condition": condition,
            "fold": fold,
            "feature": feature_name,
            "gain_importance": float(value)
        })

    return result, importance_rows


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    print("Loading:", DATA, flush=True)
    df = pd.read_csv(DATA)
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    y_raw = df[TARGET].astype(str).str.strip()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Same base feature set for both conditions.
    X_all = df.drop(columns=[TARGET, "patient_id"], errors="ignore").copy()

    # Remove missing tokens consistently, before either condition.
    X_all = X_all.replace(
        ["?", "NA", "N/A", "na", "n/a", "", "null", "None"],
        np.nan
    )

    # Convert numeric-like columns consistently.
    for col in X_all.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(X_all[col], errors="coerce")
        observed = X_all[col].notna().sum()
        if observed and converted.notna().sum() / observed >= 0.95:
            X_all[col] = converted

    if "dataset_name" not in X_all.columns:
        raise ValueError("dataset_name not found in the raw dataset.")

    X_with = X_all.copy()
    X_without = X_all.drop(columns=["dataset_name"]).copy()

    # Keep extremely rare classes in training only, consistent with the
    # existing evaluation policy. Do not report these classes as validated.
    counts = pd.Series(y).value_counts().sort_index()
    rare_codes = counts[counts < N_SPLITS].index.to_numpy()

    rare_idx = np.flatnonzero(np.isin(y, rare_codes))
    regular_idx = np.flatnonzero(~np.isin(y, rare_codes))

    X_regular = X_with.iloc[regular_idx].reset_index(drop=True)
    y_regular = y[regular_idx]

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED
    )

    rows = []
    importance_rows = []

    total = N_SPLITS * N_REPEATS

    for fold, (tr_rel, val_rel) in enumerate(
        cv.split(X_regular, y_regular), start=1
    ):
        train_idx = np.concatenate([
            regular_idx[tr_rel],
            rare_idx
        ])
        val_idx = regular_idx[val_rel]

        # Same exact rows/fold in both conditions.
        for condition, X_condition in [
            ("with_dataset_name", X_with),
            ("without_dataset_name", X_without),
        ]:
            result, imp = fit_condition(
                X_condition, y,
                train_idx, val_idx,
                condition, fold
            )
            rows.append(result)
            importance_rows.extend(imp)

            pd.DataFrame(rows).to_csv(
                OUT / "paired_cv_fold_metrics.csv",
                index=False
            )

            if importance_rows:
                pd.DataFrame(importance_rows).to_csv(
                    OUT / "gain_importance_by_fold.csv",
                    index=False
                )

    results = pd.DataFrame(rows)
    results.to_csv(
        OUT / "paired_cv_fold_metrics.csv",
        index=False
    )

    metric_cols = [
        "accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "balanced_accuracy"
    ]

    summary = results.groupby("condition")[metric_cols].agg(
        ["mean", "std"]
    )
    summary.to_csv(OUT / "paired_cv_summary.csv")

    repeat_means = results.groupby(
        ["condition", "repeat"]
    )[metric_cols].mean()
    repeat_means.to_csv(OUT / "paired_cv_repeat_means.csv")

    # Paired fold-by-fold differences, without_dataset minus with_dataset.
    wide = results.pivot(
        index="fold",
        columns="condition",
        values=metric_cols
    )

    difference_rows = []
    for metric in metric_cols:
        diff = (
            wide[(metric, "without_dataset_name")]
            - wide[(metric, "with_dataset_name")]
        )
        difference_rows.append({
            "metric": metric,
            "mean_paired_difference": float(diff.mean()),
            "sd_paired_difference": float(diff.std()),
        })

    pd.DataFrame(difference_rows).to_csv(
        OUT / "paired_cv_differences.csv",
        index=False
    )

    # Aggregate gain importance across folds.
    imp_df = pd.DataFrame(importance_rows)
    if not imp_df.empty:
        imp_summary = (
            imp_df.groupby(["condition", "feature"])["gain_importance"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(["condition", "mean"], ascending=[True, False])
        )
        imp_summary.to_csv(
            OUT / "gain_importance_summary.csv",
            index=False
        )

    with open(OUT / "metadata.json", "w") as f:
        json.dump({
            "dataset": DATA.name,
            "rows": int(len(df)),
            "classes": [str(c) for c in encoder.classes_],
            "removed_identifier": "patient_id",
            "paired_conditions": [
                "with_dataset_name",
                "without_dataset_name"
            ],
            "cv_folds": N_SPLITS,
            "cv_repeats": N_REPEATS,
            "random_state": SEED,
            "model_seed": MODEL_SEED,
            "parameters": PARAMS,
            "gpu_requested": USE_GPU,
            "rare_class_counts": {
                str(encoder.classes_[int(c)]): int(counts.loc[c])
                for c in rare_codes
            }
        }, f, indent=2)

    print("\nPAIRED CV SUMMARY", flush=True)
    print(summary.to_string(), flush=True)
    print("\nPAIRED DIFFERENCES", flush=True)
    print(pd.DataFrame(difference_rows).to_string(index=False), flush=True)
    print(f"\nFinished. Results saved to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
