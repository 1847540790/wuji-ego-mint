#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-full}"
TORCH_INDEX_URL="${MINT_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
CONDA_FALLBACK_CHANNELS="${MINT_CONDA_FALLBACK_CHANNELS:-https://conda.anaconda.org/conda-forge https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge}"
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

TEMP_CONDA_SPECS=""

cleanup_temp_specs() {
  if [[ -n "$TEMP_CONDA_SPECS" && -f "$TEMP_CONDA_SPECS" ]]; then
    find "$TEMP_CONDA_SPECS" -delete
  fi
}

trap cleanup_temp_specs EXIT

create_environment() {
  local -a create_args=(--verbose --prefix "$ENV_PREFIX" --file "$ENV_FILE")
  local -a fallback_channels=()
  local channel
  local system_channel_config

  if [[ -e "$ENV_PREFIX" ]]; then
    create_args=(--verbose --force --prefix "$ENV_PREFIX" --file "$ENV_FILE")
  fi

  system_channel_config="$(
    conda config --show channels channel_alias default_channels custom_channels 2>/dev/null || true
  )"
  if grep -Eiq 'tuna\.tsinghua\.edu\.cn|mirrors\.hit\.edu\.cn' \
    <<< "$system_channel_config"; then
    echo "[1/5] Skipping system Conda channels because a blocked TUNA or HIT source is configured"
  else
    echo "[1/5] Creating '$ENV_NAME' with the system Conda channels (Conda progress follows)"
    if conda env create "${create_args[@]}"; then
      return 0
    fi
  fi

  read -r -a fallback_channels <<< "$CONDA_FALLBACK_CHANNELS"
  if (( ${#fallback_channels[@]} == 0 )); then
    echo "System Conda channels failed and no fallback channel is configured." >&2
    return 1
  fi

  echo >&2
  echo "Trying strictly isolated fallback Conda channels." >&2
  echo "The fallback ignores configured channels and does not modify Conda settings." >&2
  TEMP_CONDA_SPECS="$(mktemp "${TMPDIR:-/tmp}/mint-conda-specs.XXXXXX")"
  sed -n 's/^  - //p' "$ENV_FILE" > "$TEMP_CONDA_SPECS"
  if [[ ! -s "$TEMP_CONDA_SPECS" ]]; then
    echo "No Conda package specifications were found in $ENV_FILE." >&2
    return 1
  fi

  for channel in "${fallback_channels[@]}"; do
    echo "Retrying Conda environment creation with: $channel" >&2
    if conda create --yes --verbose --prefix "$ENV_PREFIX" \
      --override-channels --channel "$channel" --strict-channel-priority \
      --no-default-packages --show-channel-urls --file "$TEMP_CONDA_SPECS"; then
      return 0
    fi
  done

  echo >&2
  echo "Environment creation failed with the system and fallback Conda channels." >&2
  echo "Set MINT_CONDA_FALLBACK_CHANNELS to space-separated reachable channel URLs." >&2
  return 1
}

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  if [[ "$PROFILE" == "inference" ]]; then
    echo "[1/5] Reusing existing '$ENV_NAME' environment and resuming package installation"
  else
    echo "Conda environment '$ENV_NAME' already exists." >&2
    echo "Remove it explicitly before requesting a clean rebuild:" >&2
    echo "  conda env remove --name $ENV_NAME" >&2
    exit 1
  fi
else
  if [[ -e "$ENV_PREFIX" ]]; then
    echo "[1/5] Replacing an incomplete Conda prefix at the target environment location"
  fi
  create_environment
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
