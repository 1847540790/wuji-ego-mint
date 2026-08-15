#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$PROJECT_DIR/third_party"
ENV_NAME="${MINT_ENV_NAME:-mint}"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' does not exist. Run scripts/create_env.sh full first." >&2
  exit 1
fi

cat <<'NOTICE'
MINT will download optional research software from its original repositories.

HaWoR and its checkpoints have restrictive terms, and MANO requires a separate
license agreement. Continue only if your use and redistribution plan complies
with every upstream license. Downloaded repositories remain gitignored.
NOTICE
read -r -p "Type 'I AGREE' to continue: " ACCEPTED
if [[ "$ACCEPTED" != "I AGREE" ]]; then
  echo "Installation cancelled."
  exit 1
fi

clone_component() {
  local name="$1"
  local url="$2"
  local revision="$3"
  local destination="$THIRD_PARTY_DIR/$name"
  if [[ -d "$destination/.git" ]]; then
    echo "$name already exists; leaving the checkout unchanged."
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "Refusing to overwrite non-git path: $destination" >&2
    exit 1
  fi
  git clone --filter=blob:none "$url" "$destination"
  git -C "$destination" checkout "$revision"
}

clone_component GeoCalib https://github.com/cvg/GeoCalib.git main
clone_component MoGe https://github.com/microsoft/MoGe.git main
clone_component mega-sam https://github.com/mega-sam/mega-sam.git main
clone_component HaWoR https://github.com/ThunderVVV/HaWoR.git main

conda run --name "$ENV_NAME" python -m pip install --no-deps --editable "$THIRD_PARTY_DIR/GeoCalib"
conda run --name "$ENV_NAME" python -m pip install --no-deps --editable "$THIRD_PARTY_DIR/MoGe"

cat <<'NEXT'

Source backends are installed. Model assets are intentionally not downloaded.
Follow docs/installation.md to place each checkpoint and the licensed MANO
files, then run:

  mint doctor --profile data --strict
NEXT

