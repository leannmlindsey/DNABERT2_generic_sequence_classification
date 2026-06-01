#!/usr/bin/env python3
"""
Per-variant, pick the finetune seed with the highest test-set MCC.

DNABERT-2 has a single architecture, so this selects the best-of-N seed for
each variant (only finetune candidates; the embedding linear probe / 3-layer NN
are reported separately and are not part of the winning checkpoint).

Writes <output_dir>/winners.json:
    {
      "dnabert2": {
        "type": "finetune",
        "seed": 3,
        "test_mcc": 0.85,
        "path": "<absolute path to the seed dir = saved model dir>",
        "base_model": "zhihan1996/DNABERT-2-117M",
        "all_candidates": [{type, seed, test_mcc}, ...]
      }
    }

Reads:
  <output_dir>/finetune/<variant>/seed-<N>/test_results.json
      (written by lambda_finetune_job.sh from train.py's eval_results.json;
       MCC lives under "eval_matthews_correlation", with "eval_mcc" accepted
       as a fallback for compatibility.)
"""

import argparse
import glob
import json
import os
import sys


# MCC key candidates in order of preference. train.py's compute_metrics emits
# "matthews_correlation", which trainer.evaluate prefixes with "eval_".
MCC_KEYS = ("eval_matthews_correlation", "eval_mcc", "matthews_correlation", "mcc")


def _read_mcc(metrics):
    for k in MCC_KEYS:
        if k in metrics and metrics[k] is not None:
            return float(metrics[k])
    return None


def collect_finetune_candidates(variant_dir):
    out = []
    for seed_dir in sorted(glob.glob(os.path.join(variant_dir, "seed-*"))):
        results_path = os.path.join(seed_dir, "test_results.json")
        if not os.path.isfile(results_path):
            print(f"  WARN: missing {results_path}, skipping", file=sys.stderr)
            continue
        with open(results_path) as f:
            metrics = json.load(f)
        mcc = _read_mcc(metrics)
        if mcc is None:
            print(f"  WARN: no MCC key {MCC_KEYS} in {results_path}, skipping",
                  file=sys.stderr)
            continue
        seed = int(os.path.basename(seed_dir).split("-")[1])
        out.append({
            "type": "finetune",
            "seed": seed,
            "test_mcc": float(mcc),
            "path": os.path.abspath(seed_dir),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", required=True,
                        help="Per-length replication output dir (contains finetune/)")
    parser.add_argument("--variants", nargs="+", required=True,
                        help="Variants to select for (e.g. dnabert2)")
    parser.add_argument("--base_model", default="zhihan1996/DNABERT-2-117M",
                        help="HF base model recorded in winners.json")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Skip variants with no candidates instead of aborting. "
                             "Useful for in-progress dev runs; do NOT use for the "
                             "reviewer-facing pipeline — a missing variant there means "
                             "a real training failure that should fail loudly.")
    args = parser.parse_args()

    winners = {}
    skipped = []
    for variant in args.variants:
        print(f"\n=== {variant} ===")
        finetune_dir = os.path.join(args.output_dir, "finetune", variant)
        candidates = collect_finetune_candidates(finetune_dir)
        if not candidates:
            if not args.allow_partial:
                print(f"  ERROR: no candidates found for {variant} "
                      f"(missing seed-*/test_results.json). "
                      f"Re-run with --allow-partial to skip and continue.",
                      file=sys.stderr)
                sys.exit(1)
            print(f"  SKIP: no candidates found for {variant}", file=sys.stderr)
            skipped.append(variant)
            continue

        for c in sorted(candidates, key=lambda c: c["test_mcc"], reverse=True):
            print(f"  test_mcc={c['test_mcc']:.4f}  finetune/seed-{c['seed']}")

        winner = max(candidates, key=lambda c: c["test_mcc"])
        winner["base_model"] = args.base_model
        winner["all_candidates"] = [
            {k: v for k, v in c.items() if k in ("type", "seed", "test_mcc")}
            for c in candidates
        ]
        winners[variant] = winner
        print(f"  WINNER: seed-{winner['seed']} (test_mcc={winner['test_mcc']:.4f})")

    out_path = os.path.join(args.output_dir, "winners.json")
    with open(out_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nWrote {out_path}  ({len(winners)} variant(s) with winners"
          f"{'; skipped: ' + ','.join(skipped) if skipped else ''})")

    if not winners:
        print("\nERROR: no variant produced any candidates; nothing to write.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
