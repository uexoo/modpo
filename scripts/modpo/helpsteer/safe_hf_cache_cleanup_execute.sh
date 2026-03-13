#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python3 "${REPO_ROOT}/scripts/modpo/helpsteer/utils/safe_hf_cache_cleanup.py" --execute "$@"
