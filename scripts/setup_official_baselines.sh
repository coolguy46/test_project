#!/usr/bin/env bash
set -euo pipefail

# Fetch official deep time-series baseline repositories. These repos have their
# own data formats and dependencies, so this script vendors them under
# external_baselines/ without installing anything globally.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${OUT:-${ROOT}/external_baselines}"
mkdir -p "${OUT}"
cd "${OUT}"

clone_or_update() {
  local name="$1"
  local url="$2"
  if [[ -d "${name}/.git" ]]; then
    echo "updating ${name}"
    git -C "${name}" pull --ff-only
  else
    echo "cloning ${name}"
    git clone "${url}" "${name}"
  fi
}

clone_or_update Time-Series-Library https://github.com/thuml/Time-Series-Library.git
clone_or_update TSLANet https://github.com/emadeldeen24/TSLANet.git

cat > README.md <<'EOF'
# Official External Baselines

Fetched repositories:

- `Time-Series-Library`: official home for TimesNet classification scripts.
- `TSLANet`: official ICML 2024 TSLANet implementation.

Use these repos for final-paper external deep baselines. Keep them out of the
SETM package and record exact commit hashes in the paper appendix.
EOF

git -C Time-Series-Library rev-parse HEAD > Time-Series-Library.COMMIT
git -C TSLANet rev-parse HEAD > TSLANet.COMMIT

echo "External repos ready under ${OUT}"
