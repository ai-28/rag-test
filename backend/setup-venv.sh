#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Homebrew python3.14 on some Macs breaks `venv` (ensurepip / pyexpat). Apple
# Python at /usr/bin/python3 ships with Xcode CLT and works for local dev.
PY="${PYTHON_FOR_VENV:-/usr/bin/python3}"
rm -rf .venv
"$PY" -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
echo "Done. Activate: source .venv/bin/activate"
