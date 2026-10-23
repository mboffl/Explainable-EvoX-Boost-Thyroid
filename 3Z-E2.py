# ============================================================
# 3Z-E2
# INDEPENDENT 80/20 HOLDOUT VALIDATION
# Historical Objective-1 EvoX-Boost configuration
#
# Purpose:
# Directly address reviewer concern regarding genuinely
# independent out-of-sample generalization.
#
# IMPORTANT:
# - Historical feature definition retained
# - dataset_name retained
# - TBG measured retained
# - Published EvoX-Boost parameters fixed
# - No hyperparameter optimization
# - No test-set tuning
# - Fold/training-specific preprocessing
# - No full-data final model
# - Ultra-rare classes handled explicitly
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
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
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. SETTINGS
# ============================================================

DATA_PATH = "./thyroid_cleaned.csv"

TARGET_COL = "class"

TEST_SIZE = 0.20

RANDOM_STATE = 2026

print("=" * 75)
print("3Z-E2")
print("INDEPENDENT 80/20 HOLDOUT VALIDATION")
print("Historical Objective-1 EvoX-Boost configuration")
print("=" * 75)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 3. HISTORICAL FEATURE DEFINITION
#
# IMPORTANT:
# Historical Objective-1 removed ONLY patient_id.
# dataset_name remains intentionally.
# TBG measured remains intentionally.
# ============================================================

df = df.drop(
    columns=["patient_id"],
    errors="ignore"
)

print(
    "Dropped column:",
    ["patient_id"]
)

print(
    "Historical dataset shape:",
    df.shape
)


# ============================================================
# 4. SPLIT FEATURES / TARGET
# ============================================================

X = df.drop(
    columns=[TARGET_COL]
).copy()

y_raw = df[TARGET_COL].copy()


# ============================================================
# 5. LABEL ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_raw
)

class_names = label_encoder.classes_

n_classes = len(class_names)

print(
    "\nNumber of classes:",
    n_classes
)


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

class_counts = (
    pd.Series(y)
    .value_counts()
    .sort_index()
)

print("\nClass distribution:")

for class_id in range(n_classes):

    print(
        f"{class_names[class_id]:30s}: "
        f"{class_counts.get(class_id, 0)}"
    )


# ============================================================
# 7. PROTECT ULTRA-RARE CLASSES
#
# Classes with fewer than 2 observations cannot be split.
#
# For the two classes with <5 observations, we conservatively
# keep them in training so the model has access to those
# observations, while making no claim about their independent
# test performance.
# ============================================================

protected_classes = class_counts[
    class_counts < 5
].index.tolist()

protected_mask = np.isin(
    y,
    protected_classes
)

regular_mask = ~protected_mask

print("\nProtected ultra-rare classes:")

for c in protected_classes:

    print(
        f"  {class_names[c]} "
        f"(n={class_counts[c]})"
    )


print(
    "\nRegular observations:",
    regular_mask.sum()
)

print(
    "Protected observations:",
    protected_mask.sum()
)


# ============================================================
# 8. REGULAR DATA
# ============================================================

X_regular = X.loc[
    regular_mask
].reset_index(drop=True)

y_regular = y[
    regular_mask
]

X_protected = X.loc[
    protected_mask
].reset_index(drop=True)

y_protected = y[
    protected_mask
]


# ============================================================
# 9. INDEPENDENT 80/20 STRATIFIED SPLIT
#
# ONLY regular classes are stratified.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_regular,
    y_regular,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_regular
)


# ============================================================
# 10. ADD PROTECTED OBSERVATIONS TO TRAINING
#
# They are not part of the independent test set.
# ============================================================

X_train = pd.concat(
    [
        X_train,
        X_protected
    ],
    ignore_index=True
)

y_train = np.concatenate(
    [
        y_train,
        y_protected
    ]
)


print("\n" + "=" * 75)
print("DATA PARTITION")
print("=" * 75)

print(
    "Training observations:",
    len(X_train)
)

print(
    "Independent test observations:",
    len(X_test)
)

print(
    "Total:",
    len(X_train) + len(X_test)
)

print(
    "\nIndependent test set contains:",
    len(X_test),
    "observations"
)

print(
    "Protected observations retained in training:",
    len(X_protected)
)


# ============================================================
# 11. VERIFY TEST CLASS DISTRIBUTION
# ============================================================

print("\nIndependent test class distribution:")

test_counts = (
    pd.Series(y_test)
    .value_counts()
    .sort_index()
)

for class_id in range(n_classes):

    print(
        f"{class_names[class_id]:30s}: "
        f"{test_counts.get(class_id, 0)}"
    )


# ============================================================
# 12. VERIFY NO OVERLAP
#
# Since we use row indices reset after selection, we verify
# the class counts and expected partition sizes.
# ============================================================

assert len(X_test) == 4526

print(
    "\nExpected independent test size verified: 4526"
)


# ============================================================
# 13. FEATURE TYPES
# ============================================================

numeric_features = X_train.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=[np.number]
).columns.tolist()

print(
    "\nNumeric features:",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)


# ============================================================
# 14. HISTORICAL PUBLISHED PARAMETERS
# ============================================================

PARAMS = {

    "learning_rate":
        0.0343020118,

    "max_depth":
        19,

    "subsample":
        0.99821606,

    "colsample_bytree":
        0.91857801,

    "n_estimators":
        879,

    "min_child_weight":
        1.0,

    "gamma":
        0.49494746,

    "reg_lambda":
        0.62217746
}


print("\nHistorical EvoX-Boost parameters:")

for key, value in PARAMS.items():

    print(
        f"  {key}: {value}"
    )


# ============================================================
# 15. TRAINING-ONLY PREPROCESSING
#
# CRITICAL:
# The test set is NEVER used to fit the imputer, scaler,
# or encoder.
# ============================================================

transformers = []


if numeric_features:

    numeric_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="mean"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ])

    transformers.append(
        (
            "num",
            numeric_pipeline,
            numeric_features
        )
    )


if categorical_features:

    categorical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            )
        )
    ])

    transformers.append(
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    )


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# 16. FIT PREPROCESSOR ON TRAINING DATA ONLY
# ============================================================

X_train_proc = preprocessor.fit_transform(
    X_train
)

X_test_proc = preprocessor.transform(
    X_test
)


print(
    "\nProcessed training shape:",
    X_train_proc.shape
)

print(
    "Processed test shape:",
    X_test_proc.shape
)


# ============================================================
# 17. HISTORICAL MISSING-CLASS HANDLING
#
# Check whether a class is absent from training.
# ============================================================

train_classes = np.unique(
    y_train
)

missing_classes = sorted(
    set(range(n_classes))
    - set(train_classes)
)


if missing_classes:

    print(
        "\nMissing training classes:"
    )

    for c in missing_classes:

        print(
            f"  {class_names[c]}"
        )

    dummy_X = np.zeros(
        (
            len(missing_classes),
            X_train_proc.shape[1]
        ),
        dtype=X_train_proc.dtype
    )

    dummy_y = np.array(
        missing_classes,
        dtype=y_train.dtype
    )

    X_train_proc = np.vstack(
        [
            X_train_proc,
            dummy_X
        ]
    )

    y_train_model = np.concatenate(
        [
            y_train,
            dummy_y
        ]
    )

else:

    print(
        "\nAll classes represented in training."
    )

    y_train_model = y_train


# ============================================================
# 18. XGBOOST MODEL
# ============================================================

model = XGBClassifier(

    objective="multi:softprob",

    num_class=n_classes,

    learning_rate=
        PARAMS["learning_rate"],

    max_depth=
        PARAMS["max_depth"],

    subsample=
        PARAMS["subsample"],

    colsample_bytree=
        PARAMS["colsample_bytree"],

    n_estimators=
        PARAMS["n_estimators"],

    min_child_weight=
        PARAMS["min_child_weight"],

    gamma=
        PARAMS["gamma"],

    reg_lambda=
        PARAMS["reg_lambda"],

    tree_method="hist",

    device="cuda",

    random_state=42,

    eval_metric="mlogloss",

    n_jobs=-1
)


# ============================================================
# 19. TRAIN
# ============================================================

print(
    "\nTraining historical EvoX-Boost model..."
)

model.fit(
    X_train_proc,
    y_train_model
)

print(
    "Training complete."
)


# ============================================================
# 20. INDEPENDENT TEST PREDICTION
# ============================================================

print(
    "\nGenerating predictions on untouched test set..."
)

y_pred = model.predict(
    X_test_proc
)


# ============================================================
# 21. PRIMARY METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_precision = precision_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# 22. RESULTS
# ============================================================

print("\n" + "=" * 75)
print("3Z-E2 INDEPENDENT HOLDOUT PERFORMANCE")
print("=" * 75)

print(
    f"Accuracy          : {accuracy:.6f}"
)

print(
    f"Macro Precision   : {macro_precision:.6f}"
)

print(
    f"Macro Recall      : {macro_recall:.6f}"
)

print(
    f"Macro F1          : {macro_f1:.6f}"
)

print(
    f"Weighted F1       : {weighted_f1:.6f}"
)

print(
    f"Balanced Accuracy : {balanced_accuracy:.6f}"
)


# ============================================================
# 23. PREDICTED CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("INDEPENDENT TEST PREDICTED CLASS DISTRIBUTION")
print("=" * 75)

pred_counts = (
    pd.Series(y_pred)
    .value_counts()
    .sort_index()
)

for class_id in range(n_classes):

    print(
        f"{class_names[class_id]:30s}: "
        f"{pred_counts.get(class_id, 0)}"
    )


# ============================================================
# 24. FULL CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 75)
print("INDEPENDENT TEST CLASSIFICATION REPORT")
print("=" * 75)

print(
    classification_report(
        y_test,
        y_pred,
        labels=np.arange(n_classes),
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(n_classes)
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

print("\n" + "=" * 75)
print("INDEPENDENT TEST CONFUSION MATRIX")
print("=" * 75)

print(cm_df)


# ============================================================
# 26. PER-CLASS METRICS
# ============================================================

report_dict = classification_report(
    y_test,
    y_pred,
    labels=np.arange(n_classes),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

per_class_df = pd.DataFrame(
    report_dict
).T

print("\n" + "=" * 75)
print("INDEPENDENT TEST PER-CLASS METRICS")
print("=" * 75)

print(per_class_df)


# ============================================================
# 27. SAVE RESULTS
# ============================================================

pd.DataFrame([
    {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy":
            balanced_accuracy,
        "test_size":
            len(X_test),
        "random_state":
            RANDOM_STATE
    }
]).to_csv(
    "3Z_E2_independent_holdout_results.csv",
    index=False
)


pd.DataFrame({

    "true_class_id":
        y_test,

    "predicted_class_id":
        y_pred,

    "true_label":
        [
            class_names[x]
            for x in y_test
        ],

    "predicted_label":
        [
            class_names[x]
            for x in y_pred
        ]

}).to_csv(
    "3Z_E2_independent_holdout_predictions.csv",
    index=False
)


cm_df.to_csv(
    "3Z_E2_independent_holdout_confusion_matrix.csv"
)


per_class_df.to_csv(
    "3Z_E2_independent_holdout_per_class_metrics.csv"
)


# ============================================================
# 28. FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("3Z-E2 COMPLETE")
print("=" * 75)

print(
    "\nIndependent test observations:",
    len(X_test)
)

print(
    "Protected ultra-rare observations "
    "kept exclusively in training:",
    len(X_protected)
)

print(
    "\nNo hyperparameter optimization."
)

print(
    "No test-set tuning."
)

print(
    "No full-data final model."
)

print(
    "Preprocessing fitted on training data only."
)

print(
    "\nFiles saved:"
)

print(
    "  3Z_E2_independent_holdout_results.csv"
)

print(
    "  3Z_E2_independent_holdout_predictions.csv"
)

print(
    "  3Z_E2_independent_holdout_confusion_matrix.csv"
)

print(
    "  3Z_E2_independent_holdout_per_class_metrics.csv"
)