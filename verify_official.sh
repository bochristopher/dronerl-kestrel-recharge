#!/usr/bin/env bash
# Score a checkpoint with the starter kit's own evaluator (evaluate_agent.py -> DroneRacerEvaluator),
# i.e. the same code path the AIcrowd grader uses. Must run from the starter directory because the
# evaluator loads the baseline opponents from sample_models/ by relative path.
set -euo pipefail
CKPT="$(realpath "$1")"
STARTER="${DRONERL_DIR:-$(dirname "$(realpath "$0")")/../droneRL}"
cd "$STARTER"
python evaluate_agent.py "$CKPT" 2>&1 | grep -v "it/s" | grep -E "Scores|score"
