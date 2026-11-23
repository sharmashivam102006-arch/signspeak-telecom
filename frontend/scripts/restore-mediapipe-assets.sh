#!/usr/bin/env bash
# Restores the three binary MediaPipe assets that are NOT stored in git:
#   frontend/public/mediapipe/hand_landmarker.task
#   frontend/public/mediapipe/wasm/vision_wasm_internal.wasm
#   frontend/public/mediapipe/wasm/vision_wasm_module_internal.wasm
# (plus their small .js loaders, which are stored in git.)
#
# The .wasm files ship inside the @mediapipe/tasks-vision npm package;
# the .task model file is a public download from Google.
#
# Usage: run from the repository root after `npm install` in frontend/:
#   ./frontend/scripts/restore-mediapipe-assets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PUBLIC="$ROOT/frontend/public/mediapipe"
WASM_SRC="$ROOT/frontend/node_modules/@mediapipe/tasks-vision/wasm"
TASK_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

mkdir -p "$PUBLIC/wasm"

if [ -d "$WASM_SRC" ]; then
  cp "$WASM_SRC/vision_wasm_internal.wasm" "$PUBLIC/wasm/"
  cp "$WASM_SRC/vision_wasm_module_internal.wasm" "$PUBLIC/wasm/"
  echo "Copied wasm assets from node_modules."
else
  echo "ERROR: $WASM_SRC not found. Run 'npm install' in frontend/ first." >&2
  exit 1
fi

if [ -f "$PUBLIC/hand_landmarker.task" ]; then
  echo "hand_landmarker.task already present, skipping download."
else
  curl -L -o "$PUBLIC/hand_landmarker.task" "$TASK_URL"
  echo "Downloaded hand_landmarker.task."
fi

ls -l "$PUBLIC" "$PUBLIC/wasm"
echo "MediaPipe assets restored."
