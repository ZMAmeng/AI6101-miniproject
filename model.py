import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------- Parameter ----------
MODEL_ID      = "allenai/longformer-base-4096"
MAX_LEN       = 1536
ATTN_WINDOW   = 256
tokenizer     = AutoTokenizer.from_pretrained(MODEL_ID)

# --------- Data ----------
def scrub(text: str) -> str:

    text = text.lower()
    text = re.sub(r"\S+@\S+", "<EMAIL>", text)
    text = re.sub(r"\b\d{10,}\b", "<PHONE>", text)
    return text

def tokenise_batch(batch):
    return tokenizer(
        batch["jd_clean"],
        batch["resume_clean"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

def get_model(num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=num_labels,
        attention_window=ATTN_WINDOW,
    )
