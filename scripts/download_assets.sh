#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_DIR/assets/models"
ROBOT_DIR="$PROJECT_DIR/assets/robot_hand"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
BACKBONE_URL="https://huggingface.co/robbyant/lingbot-map/resolve/main/lingbot-map.pt"
ROBOT_URL="https://github.com/1847540790/wuji-ego-mint/releases/download/assets-v0.1.0/wuji-hand-description.tar.gz"
STUDENT_HF_URL="https://huggingface.co/ZZJAsher/mint_v1/resolve/main/model.safetensors"
STUDENT_MS_REPO="AsherZhu/mint_v1"
STUDENT_MS_REVISION="master"
BACKBONE_SHA256="ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72"
ROBOT_SHA256="4594e07774211d21eda7d98ef3f7cf6f3a06f12bc750b505a8bf54765d357e69"
STUDENT_SHA256="7b2f0aa5dfd00c271bb2f12c841ccfcc70e81e4052d413eacb5fb42a1bcc36c8"

mkdir -p "$MODEL_DIR" "$ROBOT_DIR" "$CHECKPOINT_DIR"

verify() {
  local destination="$1"
  local sha256="$2"
  printf '%s  %s\n' "$sha256" "$destination" | sha256sum --check --status
}

download_verified() {
  local url="$1"
  local destination="$2"
  local sha256="$3"
  shift 3
  local partial="${destination}.part"

  if [[ -s "$destination" ]] && verify "$destination" "$sha256"; then
    echo "Verified existing asset: ${destination#"$PROJECT_DIR/"}"
    return
  fi

  echo "Downloading ${destination#"$PROJECT_DIR/"}"
  if ! curl --fail --location --retry 5 --retry-delay 2 --continue-at - \
    "$@" --output "$partial" "$url"; then
    rm -f "$partial"
    return 1
  fi
  if ! verify "$partial" "$sha256"; then
    echo "Checksum verification failed: ${partial#"$PROJECT_DIR/"}" >&2
    rm -f "$partial"
    return 1
  fi
  mv -f "$partial" "$destination"
}

download_student() {
  local destination="$CHECKPOINT_DIR/model.safetensors"
  local hf_header=()
  local hf_token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [[ -n "$hf_token" ]]; then
    hf_header=(--header "Authorization: Bearer $hf_token")
  fi

  if download_verified "$STUDENT_HF_URL" "$destination" "$STUDENT_SHA256" "${hf_header[@]}"; then
    return
  fi

  echo "Hugging Face download was unavailable; trying ModelScope."
  if ! command -v modelscope >/dev/null 2>&1; then
    echo "Install ModelScope or provide HF_TOKEN, then run this script again." >&2
    return 1
  fi

  local staging="$CHECKPOINT_DIR/.modelscope-download"
  mkdir -p "$staging"
  if ! MODELSCOPE_DOWNLOAD_PARALLELS="${MODELSCOPE_DOWNLOAD_PARALLELS:-8}" \
    modelscope download --model "$STUDENT_MS_REPO" \
    --revision "$STUDENT_MS_REVISION" \
    --local_dir "$staging" model.safetensors; then
    echo "ModelScope download failed." >&2
    rm -f "$staging/model.safetensors"
    return 1
  fi
  if ! verify "$staging/model.safetensors" "$STUDENT_SHA256"; then
    echo "ModelScope checkpoint checksum verification failed." >&2
    rm -f "$staging/model.safetensors"
    return 1
  fi
  mv -f "$staging/model.safetensors" "$destination"
}

download_verified "$BACKBONE_URL" "$MODEL_DIR/lingbot-map.pt" "$BACKBONE_SHA256"
download_verified "$ROBOT_URL" \
  "$ROBOT_DIR/wuji-hand-description.tar.gz" "$ROBOT_SHA256"
download_student

tar -xzf "$ROBOT_DIR/wuji-hand-description.tar.gz" -C "$ROBOT_DIR"
echo "Backbone, student checkpoint, and robot assets are ready."
echo "MANO is not downloaded. Follow docs/installation.md and accept its license."
