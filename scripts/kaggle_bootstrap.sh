#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/<USER>/sl_acc_reproduction.git}"
REPO_DIR="${REPO_DIR:-/kaggle/working/sl_acc_reproduction}"
RUN_CMD="${RUN_CMD:-python -m experiments.train profile=kaggle_smoke}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git pull
pip install -q -e .
pip install -q -r requirements-kaggle.txt
eval "${RUN_CMD}"
