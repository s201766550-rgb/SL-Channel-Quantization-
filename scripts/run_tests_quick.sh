#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q \
  tests/test_rounding.py \
  tests/test_acii.py \
  tests/test_cgc.py \
  tests/test_communication.py \
  tests/test_split_roundtrip.py
