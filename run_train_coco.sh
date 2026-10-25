#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gpu="${1:-0}"
run_dir="${2:-runs/rgss_coco}"
if [ "$#" -ge 2 ]; then shift 2; else set --; fi
export CUDA_VISIBLE_DEVICES="$gpu"
exec "${PYTHON:-python}" scripts/train_coco.py --run-dir "$run_dir" "$@"
