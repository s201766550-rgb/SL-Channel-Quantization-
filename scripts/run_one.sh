#!/usr/bin/env bash
set -euo pipefail
python -m experiments.train profile=kaggle_single_full "$@"
