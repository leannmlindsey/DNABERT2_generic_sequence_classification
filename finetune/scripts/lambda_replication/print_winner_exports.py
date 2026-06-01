#!/usr/bin/env python3
"""
Read winners.json[variant] and print shell-quoted export statements.
Used by lambda_inference_job.sh via:
    eval "$(python .../print_winner_exports.py <winners.json> <variant>)"
"""

import json
import shlex
import sys


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: print_winner_exports.py <winners.json> <variant>")
    winners_path, variant = sys.argv[1], sys.argv[2]
    with open(winners_path) as f:
        winners = json.load(f)
    if variant not in winners:
        sys.exit(f"ERROR: {variant} not in {winners_path}")
    w = winners[variant]
    print(f"WINNER_TYPE={shlex.quote(w.get('type', 'finetune'))}")
    print(f"WINNER_PATH={shlex.quote(w['path'])}")
    print(f"WINNER_SEED={shlex.quote(str(w.get('seed', '')))}")
    print(f"BASE_MODEL={shlex.quote(w.get('base_model', 'zhihan1996/DNABERT-2-117M'))}")


if __name__ == "__main__":
    main()
