import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import evaluate
from transformers import TrainingArguments, Trainer

from model import scrub, tokenise_batch, get_model, tokenizer

# ===== Loading Data =====
DATA_PATH   = r"Dataset\UpdatedResumeDataSet_ano.csv"
OUTPUT_DIR  = "./checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Data Processing =====
raw = pd.read_csv(DATA_PATH)
raw["jd_clean"]     = raw["Category"].apply(scrub)
raw["resume_clean"] = raw["Resume"].apply(scrub)

# ===== Create Negative Sample =====
neg= raw.copy()
neg["jd_clean"] = np.random.permutation(neg["jd_clean"].values)
neg["label"]    = 0

pos= raw.copy()
pos["label"]    = 1

df = pd.concat([pos, neg], ignore_index=True)

train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val_df,   test_df = train_test_split(tmp_df, test_size=0.50, stratify=tmp_df["label"], random_state=42)

# =====  HF Dataset & Tokenization =====
def to_ds(pdf):
    return (
        Dataset.from_pandas(pdf)
        .map(tokenise_batch, batched=True)
        .remove_columns([c for c in pdf.columns if c != "label"])
        .with_format("torch")
    )

train_ds, val_ds, test_ds = map(to_ds, (train_df, val_df, test_df))

# =====  Load model + Trainer =====
model = get_model()

metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1":       metric_f1.compute(predictions=preds, references=p.label_ids)["f1"],
    }

args = TrainingArguments(
    output_dir            = OUTPUT_DIR,
    learning_rate         = 2e-5,
    weight_decay          = 0.01,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 16,
    num_train_epochs      = 15,
    save_strategy         = "epoch",
    evaluation_strategy   = "epoch",
    report_to             = "none",
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = compute_metrics,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Test metrics:", trainer.evaluate(test_ds))
