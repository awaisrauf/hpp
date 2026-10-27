#!/usr/bin/env bash
# Download VideoCLIP-XL-v2, the retrieval model used by search_in_shots / search_in_scenes.
# The v2 repo only holds the weights; the model code (modeling.py, utils/) comes from the v1 repo.
# (Qwen3-VL and C-RADIO are fetched automatically on first use.)
#
#   bash scripts/download_models.sh [target_dir]    # default: checkpoints/VideoCLIP-XL-v2
#
# If you use a non-default target_dir, set HPP_VIDEOCLIP_DIR to it.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="${1:-checkpoints/VideoCLIP-XL-v2}"
hf download alibaba-pai/VideoCLIP-XL --include "modeling.py" --include "utils/*" --local-dir "$DIR"
hf download alibaba-pai/VideoCLIP-XL-v2 VideoCLIP-XL-v2.bin --local-dir "$DIR"
echo "Done: $DIR"
