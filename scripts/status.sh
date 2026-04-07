#!/usr/bin/env bash
# Print current project health: branch, last commit, build state, test state.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=== TDMD status ==="
echo

echo "[git]"
git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "  not a git repo yet"
git log -1 --oneline 2>/dev/null || true
echo

echo "[build]"
if [ -d build ] && [ -f build/build.ninja ]; then
  echo "  build/ exists"
  if [ -x build/tdmd_standalone ]; then
    echo "  tdmd_standalone built"
  else
    echo "  tdmd_standalone NOT built"
  fi
else
  echo "  no build/ directory; run ./scripts/build.sh"
fi
echo

echo "[milestone]"
grep -E '^\| M[0-9]' docs/03-roadmap/milestones.md 2>/dev/null | head -10 || true
echo

echo "[next steps for AI agent]"
echo "  1. Read CLAUDE.md"
echo "  2. Read docs/00-vision.md"
echo "  3. Read docs/03-roadmap/milestones.md (find current milestone)"
echo "  4. Run ./scripts/build.sh && ./scripts/run-tests.sh"
