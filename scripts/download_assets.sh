#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_DIR/assets/models"
ROBOT_DIR="$PROJECT_DIR/assets/robot_hand"
MODEL_URL="https://huggingface.co/robbyant/lingbot-map/resolve/main/lingbot-map.pt"
ROBOT_URL="https://github.com/1847540790/wuji-ego-mint/releases/download/assets-v0.1.0/wuji-hand-description.tar.gz"

mkdir -p "$MODEL_DIR" "$ROBOT_DIR"

download() {
  local url="$1"
  local destination="$2"
  if [[ -s "$destination" ]]; then
    echo "Asset already exists: $destination"
    return
  fi
  echo "Downloading $(basename "$destination")"
  curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    --output "$destination" "$url"
}

download "$MODEL_URL" "$MODEL_DIR/lingbot-map.pt"
download "$ROBOT_URL" "$ROBOT_DIR/wuji-hand-description.tar.gz"

tar -xzf "$ROBOT_DIR/wuji-hand-description.tar.gz" -C "$ROBOT_DIR"
echo "Assets are ready under assets/models and assets/robot_hand."
echo "MANO is not downloaded. Follow docs/installation.md and accept its license."
