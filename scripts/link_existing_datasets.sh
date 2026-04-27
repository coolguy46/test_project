#!/usr/bin/env bash
set -euo pipefail

# Link datasets downloaded by CLD-Trans into this project without copying them.
# Default CLD-Trans downloader target:
#   /scratch/cld-trans/datasets

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/scratch/cld-trans/datasets}"
TARGET_DATA_ROOT="${TARGET_DATA_ROOT:-${PROJECT_ROOT}/data}"

usage() {
  cat <<'EOF'
Usage: bash scripts/link_existing_datasets.sh [options]

Options:
  --source PATH       Existing dataset root. Default: /scratch/cld-trans/datasets
  --target PATH       Project data root. Default: ./data
  -h, --help          Show this help.

Example:
  bash scripts/link_existing_datasets.sh \
    --source /scratch/cld-trans/datasets \
    --target /workspace/test_project/data
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE_DATA_ROOT="$2"
      shift 2
      ;;
    --target)
      TARGET_DATA_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "${SOURCE_DATA_ROOT}" ]]; then
  echo "Source dataset root does not exist: ${SOURCE_DATA_ROOT}" >&2
  echo "Try: find /workspace /scratch -maxdepth 4 -type d -name 'ptb-xl' 2>/dev/null" >&2
  exit 1
fi

mkdir -p "${TARGET_DATA_ROOT}"

link_one() {
  local name="$1"
  local source="${SOURCE_DATA_ROOT}/${name}"
  local target="${TARGET_DATA_ROOT}/${name}"
  if [[ ! -e "${source}" ]]; then
    echo "skip missing: ${source}"
    return 0
  fi
  if [[ -L "${target}" || -e "${target}" ]]; then
    echo "exists: ${target}"
    return 0
  fi
  ln -s "${source}" "${target}"
  echo "linked: ${target} -> ${source}"
}

link_one "chb-mit"
link_one "ptb-xl"
link_one "sleep-edf"
link_one "mimic-iv-ecg"
link_one "eegmmidb"

echo "Dataset links ready under ${TARGET_DATA_ROOT}"
