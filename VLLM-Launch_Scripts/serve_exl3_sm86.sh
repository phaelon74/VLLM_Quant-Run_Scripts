#!/usr/bin/env bash
# Operator wrapper so EXL3 serve sits next to the Marlin launch script.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/Behemoth-R1-123B-v2_Compressed-Tensors/EXL3-vLLM/scripts/serve_exl3_sm86.sh" "$@"
