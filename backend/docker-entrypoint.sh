#!/bin/sh
# SignSpeak Telecom backend entrypoint.
#
# Re-downloads any AI model that is missing at container start (e.g. when
# models/ is mounted as an empty volume). The Dockerfile already downloads
# all three models at build time, so this normally does nothing.
# A failed download here only warns: the app still starts and each missing
# pipeline honestly reports itself as unavailable in GET /health instead
# of the whole container crashing.
set -e
cd /app

if [ ! -f "models/whisper-tiny/model.bin" ]; then
  echo "entrypoint: Whisper model missing, downloading..."
  python -m app.services.speech_service --download || echo "entrypoint: WARNING: Whisper download failed — ASR will report not-installed"
fi

if [ ! -f "models/piper/en_US-lessac-medium.onnx" ]; then
  echo "entrypoint: Piper voice missing, downloading..."
  python -m app.services.tts_service --download || echo "entrypoint: WARNING: Piper download failed — TTS will report not-installed"
fi

if [ ! -f "models/hand_landmarker.task" ]; then
  echo "entrypoint: MediaPipe hand_landmarker.task missing, downloading..."
  python -m app.services.vision_service --download || echo "entrypoint: WARNING: hand_landmarker download failed — sign extraction will report not-installed"
fi

exec "$@"
