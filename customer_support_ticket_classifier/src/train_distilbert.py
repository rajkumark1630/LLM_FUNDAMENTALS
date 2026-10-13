from pathlib import Path
from time import perf_counter
import json

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_NAME = "distilbert/distilbert-base-uncased"
RANDOM_STATE = 42
MAX_LENGTH = 160

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

required_columns = {"text", "category", "priority"}

for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    if not required_columns.issubset(df.columns):
        raise ValueError(f"{name}.csv is missing required columns")

    if df[list(required_columns)].isna().any().any():
        raise ValueError(f"{name}.csv contains missing values")


# --------------------------------------------------
# 3. Evaluation metrics
# --------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


# --------------------------------------------------
# 4. Train one classifier per target
# --------------------------------------------------

def train_classifier(target):
    print(f"\nTraining DistilBERT for: {target}")

    # Use training labels to define the label mapping
    classes = sorted(train_df[target].unique())
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}

    # Every split must use the same label mapping
    for name, df in [
        ("validation", val_df),
        ("test", test_df),
    ]:
        unknown = set(df[target].unique()) - set(classes)
        if unknown:
            raise ValueError(
                f"Unknown {target} labels in {name}: {unknown}"
            )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def prepare_split(df):
        frame = df[["text", target]].copy()
        frame["labels"] = frame[target].map(label2id)
        frame = frame[["text", "labels"]]

        dataset = Dataset.from_pandas(
            frame,
            preserve_index=False,
        )

        def tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=MAX_LENGTH,
            )

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=["text"],
        )

    train_ds = prepare_split(train_df)
    val_ds = prepare_split(val_df)
    test_ds = prepare_split(test_df)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = MODEL_DIR / f"distilbert_{target}"

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
        seed=RANDOM_STATE,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer
        ),
        compute_metrics=compute_metrics,
    )

    # Training time
    start = perf_counter()
    trainer.train()
    training_seconds = perf_counter() - start

    # Evaluate on validation and test sets
    validation_metrics = trainer.evaluate(
        eval_dataset=val_ds,
        metric_key_prefix="validation",
    )

    test_metrics = trainer.evaluate(
        eval_dataset=test_ds,
        metric_key_prefix="test",
    )

    # Measure average batch inference time per ticket.
    # Warm up once before timing.
    device = trainer.model.device
    trainer.model.eval()

    sample_text = test_df["text"].iloc[0]
    sample_inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    sample_inputs = {
        k: v.to(device) for k, v in sample_inputs.items()
    }

    with torch.inference_mode():
        trainer.model(**sample_inputs)

    timings = []

    for start_idx in range(0, len(test_df), 16):
        texts = test_df["text"].iloc[
            start_idx:start_idx + 16
        ].tolist()

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = perf_counter()

        with torch.inference_mode():
            trainer.model(**inputs)

        if device.type == "cuda":
            torch.cuda.synchronize()

        timings.append(perf_counter() - start)

    inference_ms = sum(timings) / len(test_df) * 1000

    # Save model and tokenizer
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "target": target,
        "classes": classes,
        "label2id": label2id,
        "id2label": id2label,
    }

    with open(output_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    result = {
        "Model": f"DistilBERT ({target})",
        "Accuracy": test_metrics["test_accuracy"],
        "Macro F1": test_metrics["test_macro_f1"],
        "Training time (seconds)": training_seconds,
        "Inference time per ticket (ms)": inference_ms,
        "Validation accuracy": validation_metrics[
            "validation_accuracy"
        ],
        "Validation macro F1": validation_metrics[
            "validation_macro_f1"
        ],
    }

    print(json.dumps(result, indent=2))
    return result


# --------------------------------------------------
# 5. Train both models and save report
# --------------------------------------------------

if __name__ == "__main__":
    results = []

    for target in ["category", "priority"]:
        results.append(train_classifier(target))

    report = pd.DataFrame(results)
    report.to_csv(
        REPORT_DIR / "distilbert_results.csv",
        index=False,
    )

    print("\nDistilBERT training completed.")
    print(report.to_string(index=False))