"""Build chat-format JSONL files for PII redaction fine-tuning.

Downloads gretelai/synthetic_pii_finance_multilingual from the Hugging Face
Hub (Apache 2.0, fully synthetic financial documents with labeled PII spans),
filters to short English documents, and converts each row into one chat
example: raw document in, redacted document plus entity list out.

Usage: python data/prepare_dataset.py
"""

import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import SYSTEM_PROMPT

DATA_DIR = Path(__file__).resolve().parent
DATASET = "gretelai/synthetic_pii_finance_multilingual"
N_TRAIN, N_VAL, N_TEST = 2500, 150, 100
MAX_DOC_CHARS = 1200
MIN_SPANS = 2
MIN_QUALITY = 80
SEED = 42


def keep(row):
    return (
        row["language"] == "English"
        and row["quality_score"] >= MIN_QUALITY
        and len(row["generated_text"]) <= MAX_DOC_CHARS
        and len(json.loads(row["pii_spans"])) >= MIN_SPANS
    )


def redact(text, spans):
    # Replace right to left so earlier span offsets stay valid.
    for s in sorted(spans, key=lambda s: s["start"], reverse=True):
        text = text[: s["start"]] + f"[{s['label'].upper()}]" + text[s["end"] :]
    return text


def to_example(row):
    text = row["generated_text"]
    spans = json.loads(row["pii_spans"])
    target = {
        "redacted_text": redact(text, spans),
        "entities": [
            {"type": s["label"], "value": text[s["start"] : s["end"]]}
            for s in sorted(spans, key=lambda s: s["start"])
        ],
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }


def main():
    ds = load_dataset(DATASET)
    rows = [r for split in ds.values() for r in split if keep(r)]
    print(f"{len(rows)} of {sum(len(s) for s in ds.values())} rows pass the filters "
          f"(English, quality >= {MIN_QUALITY}, <= {MAX_DOC_CHARS} chars, >= {MIN_SPANS} spans)")

    need = N_TRAIN + N_VAL + N_TEST
    if len(rows) < need:
        sys.exit(f"Need {need} rows but only {len(rows)} available: relax the filters")

    random.Random(SEED).shuffle(rows)
    splits = {
        "train.jsonl": rows[:N_TRAIN],
        "validation.jsonl": rows[N_TRAIN : N_TRAIN + N_VAL],
        "test.jsonl": rows[N_TRAIN + N_VAL : need],
    }
    for name, chunk in splits.items():
        path = DATA_DIR / name
        with open(path, "w") as f:
            for r in chunk:
                f.write(json.dumps(to_example(r), ensure_ascii=False) + "\n")
        print(f"{path.name}: {len(chunk)} examples")

    labels = {}
    for r in rows[:need]:
        for s in json.loads(r["pii_spans"]):
            labels[s["label"]] = labels.get(s["label"], 0) + 1
    top = ", ".join(f"{k} ({v})" for k, v in sorted(labels.items(), key=lambda kv: -kv[1])[:10])
    print(f"\n{len(labels)} PII types in the selection. Most common: {top}")


if __name__ == "__main__":
    main()
