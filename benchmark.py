"""
RockLM Benchmark Script
Dataset: deepset/prompt-injections (HuggingFace)
  - label=1 → injection/adversarial prompt (should be BLOCKED)
  - label=0 → benign prompt (should be ALLOWED)

Metrics computed:
  - Accuracy, Precision, Recall, F1
  - False Positive Rate (benign blocked), False Negative Rate (attack passed)
  - Per-category breakdown logged to logs/benchmark_results.log
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Setup benchmark logger ──────────────────────────────────────────────────
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "benchmark_results.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("benchmark")

# ── Load pipeline ───────────────────────────────────────────────────────────
logger.info("Initializing RockLM Pipeline...")
from src.core.pipeline import Pipeline
pipeline = Pipeline()

# ── Mock LLM to avoid API quota usage ───────────────────────────────────────
# We only care whether the SECURITY pipeline blocks a prompt, not the LLM response.
pipeline._call_llm = lambda prompt: "This is a safe benchmark response."
logger.info("Pipeline ready (LLM mocked — no API calls will be made).\n")


# ── Load dataset ────────────────────────────────────────────────────────────
logger.info("Loading deepset/prompt-injections dataset from HuggingFace...")
from datasets import load_dataset
ds = load_dataset("deepset/prompt-injections", split="train")
logger.info(f"Dataset loaded: {len(ds)} samples\n")

# ── Run benchmark ───────────────────────────────────────────────────────────
BLOCKED_RESPONSE = "I apologize, but I cannot process that request."

results = []
tp = tn = fp = fn = 0
total = len(ds)

logger.info(f"{'='*60}")
logger.info(f"Starting benchmark on {total} prompts")
logger.info(f"{'='*60}\n")

for i, sample in enumerate(ds):
    prompt = sample["text"]
    true_label = sample["label"]   # 1=injection, 0=benign
    user_id = f"benchmark_user_{i % 50}"  # rotating users — context builds up for drift detection

    start = time.time()
    try:
        response = pipeline.run(user_id, prompt)
    except Exception as e:
        response = f"ERROR: {e}"
    elapsed = round(time.time() - start, 3)

    # Determine if pipeline blocked it
    was_blocked = (response == BLOCKED_RESPONSE)

    # true_label=1 → attack → expected: blocked
    # true_label=0 → benign → expected: allowed
    if true_label == 1 and was_blocked:
        outcome = "TP"
        tp += 1
    elif true_label == 0 and not was_blocked:
        outcome = "TN"
        tn += 1
    elif true_label == 0 and was_blocked:
        outcome = "FP"
        fp += 1
    elif true_label == 1 and not was_blocked:
        outcome = "FN"
        fn += 1

    result = {
        "index": i,
        "user_id": user_id,
        "prompt": prompt[:120],
        "true_label": "INJECTION" if true_label == 1 else "BENIGN",
        "was_blocked": was_blocked,
        "outcome": outcome,
        "latency_s": elapsed,
    }
    results.append(result)

    logger.info(
        f"[{i+1:>4}/{total}] [{outcome}] label={'INJ' if true_label==1 else 'BEN'} "
        f"blocked={was_blocked} latency={elapsed}s | {prompt[:60]!r}"
    )

# ── Compute metrics ─────────────────────────────────────────────────────────
accuracy  = (tp + tn) / total if total else 0
precision = tp / (tp + fp) if (tp + fp) else 0
recall    = tp / (tp + fn) if (tp + fn) else 0
f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
fpr       = fp / (fp + tn) if (fp + tn) else 0  # false positive rate
fnr       = fn / (fn + tp) if (fn + tp) else 0  # false negative rate
avg_latency = sum(r["latency_s"] for r in results) / len(results)

summary = {
    "dataset": "deepset/prompt-injections",
    "timestamp": datetime.now().isoformat(),
    "total_samples": total,
    "true_positives": tp,
    "true_negatives": tn,
    "false_positives": fp,
    "false_negatives": fn,
    "accuracy":  round(accuracy,  4),
    "precision": round(precision, 4),
    "recall":    round(recall,    4),
    "f1_score":  round(f1,        4),
    "false_positive_rate": round(fpr, 4),
    "false_negative_rate": round(fnr, 4),
    "avg_latency_s": round(avg_latency, 3),
}

logger.info("\n" + "="*60)
logger.info("BENCHMARK RESULTS SUMMARY")
logger.info("="*60)
for k, v in summary.items():
    logger.info(f"  {k:<28}: {v}")
logger.info("="*60)

# Save full results to JSON
json_out = log_dir / "benchmark_results.json"
with open(json_out, "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)

logger.info(f"\nFull results saved to: {json_out}")
logger.info(f"Log saved to:          {log_file}")
