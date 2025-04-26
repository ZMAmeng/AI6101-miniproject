import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from model import scrub
CHECKPOINT_DIR = "./checkpoints"

# Loading Model / tokenizer
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR)
sep_token = tokenizer.sep_token

pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0,         # GPU:0  CPU:-1
)

def rank_resumes(jd_text: str, resume_list, top_k: int = 5):
    jd_text = scrub(jd_text)
    scored  = []
    for r in resume_list:
        pred = pipe(
            jd_text + sep_token + scrub(r),
            truncation=True,
            max_length=512,
        )[0]
        scored.append((pred["score"], r))
    return sorted(scored, reverse=True)[:top_k]

# --- CLI Test ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jd",  required=True, help="a Job Description")
    parser.add_argument("--cv",  nargs="+",   required=True, help="Several resume text paths")
    parser.add_argument("-k",    type=int, default=5, help="Top-k Return Numbers")
    args = parser.parse_args()

    resumes = [open(p, encoding="utf8").read() for p in args.cv]
    for score, res in rank_resumes(args.jd, resumes, top_k=args.k):
        print(f"{score:.4f} | {res[:80]}…")
