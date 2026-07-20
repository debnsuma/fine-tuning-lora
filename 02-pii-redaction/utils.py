"""Helpers for the PII redaction fine-tuning notebook.

Everything that is not the core fine-tune / deploy / inference workflow
lives here: dataset validation, training curve plotting, and entity level
evaluation. The notebook imports these so its cells stay short.
"""

import json
import re

import matplotlib.pyplot as plt

# Single source of truth for the task prompt. prepare_dataset.py bakes it
# into every training example, and inference must send the exact same text.
SYSTEM_PROMPT = (
    "You are a PII redaction engine for financial documents. Find every span "
    "of personally identifiable information in the document: names, companies, "
    "dates, street addresses, account or routing numbers, emails, phone "
    "numbers, and any other identifier tied to a person or organization. "
    'Respond with a single JSON object with two keys: "redacted_text", the '
    "full document with each PII span replaced by its type in uppercase "
    'square brackets such as [NAME] or [EMAIL], and "entities", a list of '
    '{"type": ..., "value": ...} objects, one per span in order of '
    "appearance. Respond with only the JSON object."
)

VALID_ROLES = {"system", "user", "assistant"}

# Chart tokens: recessive greys for chrome, two categorical hues for series.
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_BLUE = "#2a78d6"
SERIES_GREEN = "#008300"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def validate_dataset(path):
    """Check a chat JSONL file against the rules the fine-tuning service
    enforces, plus this task's own output contract. Prints a one line
    verdict (with details when something is wrong) and returns True/False.
    """
    problems = []
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                problems.append(f"line {i}: not valid JSON")

    for i, row in enumerate(rows):
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            problems.append(f"row {i}: missing or empty 'messages'")
            continue
        for j, m in enumerate(msgs):
            if m.get("role") not in VALID_ROLES:
                problems.append(f"row {i}, message {j}: unknown role {m.get('role')!r}")
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                problems.append(f"row {i}, message {j}: empty content")
        if msgs[-1].get("role") != "assistant":
            problems.append(f"row {i}: last turn must be 'assistant', that is where the loss is")
            continue
        try:
            target = json.loads(msgs[-1]["content"])
            assert set(target) == {"redacted_text", "entities"}
            assert all(set(e) == {"type", "value"} for e in target["entities"])
        except (json.JSONDecodeError, AssertionError, TypeError):
            problems.append(f"row {i}: assistant content is not the expected JSON contract")

    if problems:
        print(f"{path}: {len(rows)} rows, {len(problems)} problems")
        for p in problems[:10]:
            print("   ", p)
        if len(problems) > 10:
            print(f"    ... and {len(problems) - 10} more")
        return False

    tokens = sum(len(m["content"]) for r in rows for m in r["messages"]) // 4
    print(f"{path}: {len(rows)} rows, all checks passed (~{tokens:,} tokens, chars/4 heuristic)")
    return True


def _style_axes(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_training_curves(payload, title="LoRA fine-tune, loss curves"):
    """Render train/validation loss from the /metrics endpoint payload."""
    series = {m["metric_name"]: m["data"] for m in payload.get("metrics", [])}
    curves = [
        ("ev_loss", "Training loss", SERIES_BLUE),
        ("ev_eval_loss", "Validation loss", SERIES_GREEN),
    ]
    present = [(n, l, c) for n, l, c in curves if series.get(n)]
    if not present:
        print("No loss series yet, re-run this cell once the job is running.")
        return

    t0 = min(series[n][0]["x"] for n, _, _ in present)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=110)

    for name, label, color in present:
        pts = series[name]
        xs = [(p["x"] - t0) / 60_000 for p in pts]
        ys = [p["value"] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=2.2, label=label,
                marker="o", markersize=3.5, markevery=max(1, len(xs) // 12))
        if name == "ev_loss":
            ax.fill_between(xs, ys, min(ys) * 0.98, color=color, alpha=0.06)
        ax.plot(xs[-1], ys[-1], "o", color=color, markersize=6)
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)

    if series.get("ev_eval_loss"):
        pts = series["ev_eval_loss"]
        best = min(pts, key=lambda p: p["value"])
        bx, by = (best["x"] - t0) / 60_000, best["value"]
        ax.plot(bx, by, "o", markersize=10, markerfacecolor="none",
                markeredgecolor=SERIES_GREEN, markeredgewidth=1.5)
        ax.annotate(f"best {by:.3f}", (bx, by), xytext=(-6, 12),
                    textcoords="offset points", ha="right", fontsize=9, color=INK_SOFT)

    _style_axes(ax)
    ax.set_xlabel("Minutes since training started", color=INK_SOFT)
    ax.set_ylabel("Loss", color=INK_SOFT)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=22)
    ax.legend(frameon=False, loc="upper right", fontsize=9)

    tokens = series.get("ev_num_input_tokens_seen")
    if tokens:
        ax.text(0, 1.03, f"{int(tokens[-1]['value']):,} training tokens processed",
                transform=ax.transAxes, fontsize=9, color=INK_MUTED)
    plt.tight_layout()
    plt.show()


def parse_prediction(reply):
    """Extract the JSON prediction from a model reply. Returns a dict with
    'redacted_text' and 'entities', or None when the reply is unusable."""
    text = (reply or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        entities = obj.get("entities")
        assert isinstance(entities, list)
        obj["entities"] = [
            {"type": str(e.get("type", "")), "value": str(e.get("value", ""))}
            for e in entities if isinstance(e, dict)
        ]
        return obj
    except (json.JSONDecodeError, AssertionError):
        return None


def _norm(value):
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def score_entities(reference, predicted):
    """Entity level scores at two strictness levels.

    detection: the PII string was found, whatever type it was given
    typed: both the string and its type label match the reference
    """
    scores = {}
    for level in ("detection", "typed"):
        if level == "detection":
            ref = {_norm(e["value"]) for e in reference}
            pred = {_norm(e["value"]) for e in predicted}
        else:
            ref = {(_norm(e["type"]), _norm(e["value"])) for e in reference}
            pred = {(_norm(e["type"]), _norm(e["value"])) for e in predicted}
        tp = len(ref & pred)
        scores[level] = {"tp": tp, "fp": len(pred - ref), "fn": len(ref - pred)}
    return scores


def aggregate_scores(rows):
    """Micro-averaged precision/recall/F1 over per-example score dicts."""
    out = {}
    for level in ("detection", "typed"):
        tp = sum(r[level]["tp"] for r in rows)
        fp = sum(r[level]["fp"] for r in rows)
        fn = sum(r[level]["fn"] for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out[level] = {"precision": p, "recall": r, "f1": f1}
    return out


def report_eval(summaries, parse_failures, n_examples):
    """Print the comparison table and render the F1 chart.

    summaries: {model_label: aggregate_scores(...) result}
    parse_failures: {model_label: count of unparseable replies}
    """
    print(f"Entity level scores on {n_examples} held-out documents\n")
    header = f"{'Model':<38} {'Det P':>7} {'Det R':>7} {'Det F1':>7} {'Typed F1':>9} {'Bad JSON':>9}"
    print(header)
    print("-" * len(header))
    for label, s in summaries.items():
        d, t = s["detection"], s["typed"]
        print(f"{label:<38} {d['precision']:>6.0%} {d['recall']:>6.0%} "
              f"{d['f1']:>6.0%} {t['f1']:>8.0%} {parse_failures.get(label, 0):>9}")

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    metrics = ["Detection F1", "Typed F1"]
    width = 0.36
    colors = [SERIES_BLUE, SERIES_GREEN]
    for i, (label, s) in enumerate(summaries.items()):
        vals = [s["detection"]["f1"] * 100, s["typed"]["f1"] * 100]
        bars = ax.bar([x + (i - 0.5) * width for x in range(len(metrics))],
                      vals, width * 0.94, label=label, color=colors[i % len(colors)])
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}%", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=10, color=INK)
    _style_axes(ax)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, color=INK_SOFT)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Micro-averaged F1", color=INK_SOFT)
    ax.set_title(f"PII extraction quality on {n_examples} held-out documents",
                 loc="left", color=INK, fontsize=12, pad=10)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()
