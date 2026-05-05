#!/usr/bin/env bash
set -euo pipefail
python -m experiments.train profile=kaggle_matrix_shard "$@"
