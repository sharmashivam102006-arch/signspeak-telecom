.PHONY: demo up down api web synth train eval export smoke parity verify clean

## Run the zero-setup demo (no install required)
demo:
	@echo "→ http://localhost:8080/demo/"
	@python3 -m http.server 8080

up:
	docker compose up -d postgres redis livekit

down:
	docker compose down

api:
	cd apps/api && uvicorn signspeak_api.main:app --reload --port 8000

web:
	cd apps/web && npm run dev

## Generate a synthetic dataset so the pipeline runs before you own a camera
synth:
	cd ml && python scripts/make_synthetic.py --subjects 6 --per-class 20

train:
	cd ml && python -m signspeak_ml.train --data data/raw --compare

eval:
	cd ml && python -m signspeak_ml.evaluate --data data/raw --checkpoint artifacts/gru.pt --baseline

export:
	cd ml && python -m signspeak_ml.export_onnx --checkpoint artifacts/gru.pt --out ../apps/web/public/models

## End-to-end plumbing check on synthetic data
smoke:
	cd ml && python scripts/smoke_test.py

## Assert the Python and TypeScript feature encoders agree (run this in CI)
parity:
	cd ml && python scripts/check_parity.py

verify: smoke parity

clean:
	rm -rf ml/data/raw/synthetic* ml/artifacts/*.pt ml/artifacts/*.json
