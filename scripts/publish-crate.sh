#!/usr/bin/env bash
set -euo pipefail

dry_run="${DRY_RUN_ONLY:-false}"

if [[ "$dry_run" == "true" ]]; then
  cargo publish --dry-run
  exit 0
fi

: "${CRATES_IO_API_TOKEN:?CRATES_IO_API_TOKEN is required for a real publish}"

cargo publish --token "$CRATES_IO_API_TOKEN"
