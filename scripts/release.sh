#!/usr/bin/env bash
# release.sh - Build, tag, and publish aceteam-aep to PyPI
#
# Usage:
#   ./scripts/release.sh -v 0.2.0              # Release version 0.2.0
#   ./scripts/release.sh -v 0.2.0 --dry-run    # Preview without publishing
#   ./scripts/release.sh -v 0.2.0 --no-tag     # Publish without git tag
#
set -euo pipefail

VERSION=""
DRY_RUN=false
NO_TAG=false

usage() {
  echo "Usage: $0 -v VERSION [--dry-run] [--no-tag]"
  echo ""
  echo "Options:"
  echo "  -v, --version VERSION   Version to release (required, e.g. 0.2.0)"
  echo "  --dry-run               Preview actions without executing"
  echo "  --no-tag                Skip git tag creation"
  echo "  -h, --help              Show this help"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -v|--version) VERSION="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --no-tag) NO_TAG=true; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

if [[ -z "$VERSION" ]]; then
  echo "Error: version is required"
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Verify we're on a clean working tree
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Error: working tree is not clean. Commit or stash changes first."
  exit 1
fi

# Check current version in pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
echo "Current version: $CURRENT_VERSION"
echo "New version:     $VERSION"

if [[ "$CURRENT_VERSION" == "$VERSION" ]]; then
  echo "Version already set to $VERSION in pyproject.toml"
else
  if $DRY_RUN; then
    echo "[dry-run] Would update pyproject.toml version to $VERSION"
  else
    sed -i "s/^version = \"$CURRENT_VERSION\"/version = \"$VERSION\"/" pyproject.toml
    echo "Updated pyproject.toml to version $VERSION"
  fi
fi

# Build
echo ""
echo "Building..."
if $DRY_RUN; then
  echo "[dry-run] Would run: uv build"
else
  uv build
fi

# Publish
echo ""
echo "Publishing to PyPI..."
if $DRY_RUN; then
  echo "[dry-run] Would run: uv publish"
else
  uv publish
fi

# Tag
if ! $NO_TAG; then
  echo ""
  echo "Creating git tag v$VERSION..."
  if $DRY_RUN; then
    echo "[dry-run] Would run: git add pyproject.toml && git commit && git tag v$VERSION && git push"
  else
    if [[ "$CURRENT_VERSION" != "$VERSION" ]]; then
      git add pyproject.toml
      git commit -m "release: v$VERSION"
    fi
    git tag -a "v$VERSION" -m "Release v$VERSION"
    git push origin main "v$VERSION"
  fi
fi

echo ""
echo "Done! Published aceteam-aep $VERSION to PyPI"
echo "https://pypi.org/project/aceteam-aep/$VERSION/"
