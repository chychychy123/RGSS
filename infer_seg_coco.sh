#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gpu="${1:-0}"
run_dir="${2:-runs/rgss_coco}"
output_dir="${3:-results/rgss_coco}"
if [ "$#" -ge 3 ]; then shift 3; else set --; fi
export CUDA_VISIBLE_DEVICES="$gpu"
exec "${PYTHON:-python}" scripts/eval_coco.py --run "$run_dir" --out "$output_dir" "$@"
