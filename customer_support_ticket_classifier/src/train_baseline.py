from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGETS = ["category", "priority"]


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")

required_columns = {"text", "category", "priority"}

for name, frame in [
    ("train.csv", train_df),
    ("validation.csv", val_df),
]:
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

    if frame[list(required_columns)].isna().any().any():
        raise ValueError(f"{name} contains missing required values.")

X_train = train_df["text"].astype(str)
X_val = val_df["text"].astype(str)

print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))


# --------------------------------------------------
# 3. Define word-only and word + character pipelines
# --------------------------------------------------

def build_word_pipeline():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                sublinear_tf=True,
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_word_char_pipeline():
    features = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=True,
                sublinear_tf=True,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


# --------------------------------------------------
# 4. Hyperparameter tuning with cross-validation
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

word_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}

word_char_grid = {
    "features__word__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.1, 1.0, 10.0],
}

results = []
best_models = {}

for target in TARGETS:
    print(f"\n{'=' * 60}")
    print(f"TARGET: {target.upper()}")
    print("=" * 60)

    y_train = train_df[target].astype(str)
    y_val = val_df[target].astype(str)

    # Word-only baseline
    searches = {
        "word": GridSearchCV(
            estimator=build_word_pipeline(),
            param_grid=word_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
        "word_char": GridSearchCV(
            estimator=build_word_char_pipeline(),
            param_grid=word_char_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        ),
    }

    target_results = []

    for feature_name, search in searches.items():
        print(f"\nTraining model: {feature_name}")

        search.fit(X_train, y_train)

        model = search.best_estimator_
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        print("Best parameters:", search.best_params_)
        print("Best CV macro F1:", round(search.best_score_, 4))
        print("Validation accuracy:", round(accuracy, 4))
        print("Validation macro F1:", round(macro_f1, 4))

        print("\nClassification report:")
        print(classification_report(
            y_val,
            predictions,
            zero_division=0,
        ))

        labels = sorted(y_train.unique())

        matrix = confusion_matrix(
            y_val,
            predictions,
            labels=labels,
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        )

        matrix_path = (
            REPORT_DIR
            / f"{target}_{feature_name}_confusion_matrix.csv"
        )

        matrix_df.to_csv(matrix_path)

        print("Confusion matrix:")
        print(matrix_df)

        results.append({
            "target": target,
            "features": feature_name,
            "cv_macro_f1": search.best_score_,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "best_params": str(search.best_params_),
        })

        target_results.append({
            "features": feature_name,
            "model": model,
            "cv_macro_f1": search.best_score_,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

        # Save both the baseline and bonus models
        model_path = (
            MODEL_DIR / f"{target}_{feature_name}.joblib"
        )

        joblib.dump(model, model_path)
        print("Saved model:", model_path)

    # Pick the model with the best cross-validation macro F1
    target_results.sort(
        key=lambda item: item["cv_macro_f1"],
        reverse=True,
    )

    best = target_results[0]

    best_models[target] = best["model"]

    joblib.dump(
        best["model"],
        MODEL_DIR / f"{target}_best.joblib",
    )

    print(f"\nBest feature setup for {target}: {best['features']}")


# --------------------------------------------------
# 5. Save metrics comparison
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = REPORT_DIR / "baseline_metrics.csv"
results_df.to_csv(results_path, index=False)

print("\nMETRICS COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved metrics:", results_path)


# --------------------------------------------------
# 6. Error analysis: 10 most confusing misclassified
#    validation examples for each target
# --------------------------------------------------

error_records = []

for target in TARGETS:
    model = best_models[target]
    y_true = val_df[target].astype(str).reset_index(drop=True)
    predictions = model.predict(X_val)

    probabilities = model.predict_proba(X_val)
    classes = list(model.classes_)

    for i, (actual, predicted) in enumerate(
        zip(y_true, predictions)
    ):
        if actual == predicted:
            continue

        # Difference between the two highest probabilities.
        # A small margin means the prediction was uncertain.
        sorted_probs = sorted(probabilities[i], reverse=True)

        margin = (
            sorted_probs[0] - sorted_probs[1]
            if len(sorted_probs) > 1
            else sorted_probs[0]
        )

        predicted_probability = probabilities[i][
            classes.index(predicted)
        ]

        error_records.append({
            "target": target,
            "text": X_val.iloc[i],
            "actual": actual,
            "predicted": predicted,
            "predicted_probability": predicted_probability,
            "probability_margin": margin,
            "guess_why": (
                "Possible overlap in wording, insufficient context, "
                "or limited examples in the synthetic dataset."
            ),
        })

errors_df = pd.DataFrame(error_records)

if not errors_df.empty:
    errors_df = errors_df.sort_values(
        "probability_margin",
        ascending=True,
    )

    # Up to 10 confusing examples for each label
    for target in TARGETS:
        target_errors = errors_df[
            errors_df["target"] == target
        ].head(10)

        print(f"\nTOP CONFUSING EXAMPLES: {target.upper()}")
        print(target_errors[
            [
                "text",
                "actual",
                "predicted",
                "predicted_probability",
                "guess_why",
            ]
        ].to_string(index=False))

    errors_df.groupby("target", group_keys=False).head(10).to_csv(
        REPORT_DIR / "top_confusing_examples.csv",
        index=False,
    )
else:
    print(
        "\nNo validation errors were found. "
        "The error-analysis file was not created."
    )


print("\nTask 2 completed successfully!")
print("Models folder:", MODEL_DIR)
print("Reports folder:", REPORT_DIR)