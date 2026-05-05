#!/usr/bin/env bash
set -euo pipefail
cd /kaggle/working
zip -r sl_acc_outputs.zip outputs
echo "Created /kaggle/working/sl_acc_outputs.zip"
