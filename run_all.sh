#!/usr/bin/env bash
# Runs every system one after the other. Each run writes its figures to results/<system>/.
# Usage: bash run_all.sh [extra run.py options, e.g. --seeds 10 --workers 16]
set -euo pipefail
cd "$(dirname "$0")"
for sys in pendulum cartpole launch flexarm flexarm3; do
    python run.py --system "$sys" --out "results/$sys" "$@"
done
