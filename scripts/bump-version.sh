#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo 'Usage: scripts/bump-version.sh <version|--rc|--patch|--minor|--major>' >&2
  exit 2
fi

cd "$(dirname "$0")/.."
python3 scripts/release_plan.py bump -- "$1"
uv lock
python3 scripts/release_plan.py check
echo 'Version files and uv.lock updated. Review and commit them in a PR.'
