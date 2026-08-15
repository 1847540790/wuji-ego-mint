#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-full}"
TORCH_INDEX_URL="${MINT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
PIP_NETWORK_ARGS=(--retries 10 --timeout 120)
PYPI_SOURCE_ARGS=()
if [[ -n "${MINT_PYPI_INDEX_URL:-}" ]]; then
  PYPI_SOURCE_ARGS=(--index-url "$MINT_PYPI_INDEX_URL")
fi
TORCH_SOURCE_ARGS=(--extra-index-url "$TORCH_INDEX_URL")
if [[ -n "${MINT_TORCH_FIND_LINKS:-}" ]]; then
  TORCH_SOURCE_ARGS=(--find-links "$MINT_TORCH_FIND_LINKS")
fi

case "$PROFILE" in
  full)
    ENV_NAME="mint"
    ENV_FILE="$PROJECT_DIR/environments/mint.yml"
    REQUIREMENTS=(
      requirements-inference.txt
      requirements-train.txt
      requirements-data.txt
      requirements-dev.txt
    )
    DOCTOR_PROFILE="full"
    ;;
  inference)
    ENV_NAME="mint-inference"
    ENV_FILE="$PROJECT_DIR/environments/mint-inference.yml"
    REQUIREMENTS=(requirements-inference.txt)
    DOCTOR_PROFILE="inference"
    ;;
  *)
    echo "Usage: $0 {full|inference}" >&2
    exit 2
    ;;
esac

ENV_PREFIX="$(conda info --json | python -c '
import json
import os
import sys

info = json.load(sys.stdin)
print(os.path.join(info["envs_dirs"][0], sys.argv[1]))
' "$ENV_NAME")"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME" || [[ -e "$ENV_PREFIX" ]]; then
  if [[ "$PROFILE" == "inference" ]]; then
    echo "[1/5] Reusing existing '$ENV_NAME' environment and resuming package installation"
  else
    echo "Conda environment '$ENV_NAME' already exists." >&2
    echo "Remove it explicitly before requesting a clean rebuild:" >&2
    echo "  conda env remove --name $ENV_NAME" >&2
    exit 1
  fi
else
  echo "[1/5] Creating '$ENV_NAME' from $ENV_FILE (Conda progress follows)"
  conda env create --verbose --file "$ENV_FILE"
fi

echo "[2/5] Installing the tested CUDA-enabled PyTorch build (large CUDA wheels)"
conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
  --progress-bar on \
  "${PIP_NETWORK_ARGS[@]}" "${PYPI_SOURCE_ARGS[@]}" "${TORCH_SOURCE_ARGS[@]}" \
  --requirement "$PROJECT_DIR/environments/requirements-torch.txt"

for requirement in "${REQUIREMENTS[@]}"; do
  echo "[3/5] Installing $requirement"
  conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
    --progress-bar on \
    "${PIP_NETWORK_ARGS[@]}" "${PYPI_SOURCE_ARGS[@]}" \
    --requirement "$PROJECT_DIR/environments/$requirement"
done

echo "[4/5] Installing MINT in editable mode without dependency re-resolution"
conda run --no-capture-output --name "$ENV_NAME" python -m pip install \
  --progress-bar on --no-deps --editable "$PROJECT_DIR"

echo "[5/5] Running the environment smoke test"
conda run --no-capture-output --cwd "$PROJECT_DIR" --name "$ENV_NAME" \
  python -m mint doctor --profile "$DOCTOR_PROFILE" --strict

echo
echo "Environment '$ENV_NAME' is ready."
echo "Activate it with: conda activate $ENV_NAME"
