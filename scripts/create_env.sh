#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-full}"

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

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists." >&2
  echo "Remove it explicitly before requesting a clean rebuild:" >&2
  echo "  conda env remove --name $ENV_NAME" >&2
  exit 1
fi

echo "Creating '$ENV_NAME' from $ENV_FILE"
conda env create --file "$ENV_FILE"

echo "Installing the tested CUDA-enabled PyTorch build"
conda run --name "$ENV_NAME" python -m pip install \
  --requirement "$PROJECT_DIR/environments/requirements-torch.txt"

for requirement in "${REQUIREMENTS[@]}"; do
  echo "Installing $requirement"
  conda run --name "$ENV_NAME" python -m pip install \
    --requirement "$PROJECT_DIR/environments/$requirement"
done

echo "Installing MINT in editable mode without dependency re-resolution"
conda run --name "$ENV_NAME" python -m pip install --no-deps --editable "$PROJECT_DIR"

echo "Running the environment smoke test"
conda run --name "$ENV_NAME" mint doctor --profile "$DOCTOR_PROFILE" --strict

echo
echo "Environment '$ENV_NAME' is ready."
echo "Activate it with: conda activate $ENV_NAME"

