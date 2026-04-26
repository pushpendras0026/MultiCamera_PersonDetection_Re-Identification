"""
run_evaluation.py
-----------------
Run all ReID benchmark evaluations and produce a single combined
comparison report in results/benchmark_comparison.json and .csv.

Usage:
    python run_evaluation.py [--ilids] [--market] [--all]

Default (no flags) = --all
"""

import argparse, json, csv, sys, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR  = PROJECT_ROOT / "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--ilids",  action="store_true", help="Run i-LIDS-VID evaluation")
parser.add_argument("--market", action="store_true", help="Run Market-1501 evaluation")
parser.add_argument("--all",    action="store_true", help="Run all evaluations (default)")
args = parser.parse_args()

run_all = args.all or (not args.ilids and not args.market)

# ── Lazy import REID once ────────────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from reid import REID

print("Loading REID model (ResNet50) ...")
reid = REID()
print("REID model loaded.\n")

all_results = {}

# ── i-LIDS-VID ───────────────────────────────────────────────────────────────
if run_all or args.ilids:
    from evaluate_ilids import evaluate_ilids
    scores = evaluate_ilids(reid)
    all_results["i-LIDS-VID"] = scores

# ── Market-1501 ───────────────────────────────────────────────────────────────
if run_all or args.market:
    from evaluate_market1501 import evaluate_market1501
    scores = evaluate_market1501(reid)
    if scores:
        all_results["Market-1501"] = scores

# ── Combined comparison table ─────────────────────────────────────────────────
RANKS = [1, 5, 10, 20]

print("\n")
print("=" * 80)
print("  BENCHMARK COMPARISON — Your Pipeline vs Published State-of-the-Art")
print("=" * 80)

header = f"{'Method':<38} {'Dataset':<14} {'Rank-1':>7} {'Rank-5':>7} {'Rank-10':>8} {'mAP':>7}"
print(header)
print("-" * 80)

rows = []  # for CSV

for dataset_name, scores in all_results.items():
    if scores is None:
        continue

    # Your pipeline row
    r1  = scores.get("Rank-1",  "N/A")
    r5  = scores.get("Rank-5",  "N/A")
    r10 = scores.get("Rank-10", "N/A")
    map_ = scores.get("mAP",    "N/A")
    line = f"{'★ YOUR PIPELINE (ResNet50+EMA)':<38} {dataset_name:<14} {r1:>7} {r5:>7} {r10:>8} {map_:>7}"
    print(line)
    rows.append({
        "Method": "YOUR PIPELINE (ResNet50+EMA)",
        "Dataset": dataset_name,
        "Rank-1": r1, "Rank-5": r5, "Rank-10": r10, "mAP": map_
    })

    # Baselines
    for bname, binfo in scores.get("published_baselines", {}).items():
        br1  = binfo.get("Rank-1",  "N/A")
        br5  = binfo.get("Rank-5",  "N/A")
        br10 = binfo.get("Rank-10", "N/A")
        bmap = binfo.get("mAP",     "N/A")
        line = f"  {bname:<36} {dataset_name:<14} {str(br1):>7} {str(br5):>7} {str(br10):>8} {str(bmap):>7}"
        print(line)
        rows.append({
            "Method": bname,
            "Dataset": dataset_name,
            "Rank-1": br1, "Rank-5": br5, "Rank-10": br10, "mAP": bmap
        })
    print()

print("=" * 80)

# ── Save combined JSON ────────────────────────────────────────────────────────
combined = {
    "all_results":   all_results,
    "comparison_rows": rows,
}
json_out = RESULTS_DIR / "benchmark_comparison.json"
with open(json_out, "w") as f:
    json.dump(combined, f, indent=2)
print(f"\n[Saved] {json_out}")

# ── Save comparison CSV ───────────────────────────────────────────────────────
csv_out = RESULTS_DIR / "benchmark_comparison.csv"
if rows:
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "Dataset", "Rank-1", "Rank-5", "Rank-10", "mAP"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {csv_out}")

print("\nDone.")
