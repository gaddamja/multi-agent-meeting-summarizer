PYTHON?=python3
VENV=.venv

.PHONY: setup transcribe clean docker-build run-gradio test

setup:
	@echo "Running automated setup..."
	@chmod +x scripts/setup.sh
	./scripts/setup.sh

transcribe:
	@echo "Run transcription on included sample audio"
	@if [ -f amicorpus/IS1002b/audio/IS1002b.Mix-Headset.wav ]; then \
		. $(VENV)/bin/activate && python scripts/run_transcription.py amicorpus/IS1002b/audio/IS1002b.Mix-Headset.wav --model small --output sample_audio_out.json || true; \
	else \
		echo "Sample audio not found. Add an audio file to amicorpus/ or specify path: make transcribe FILE=your_file.wav"; \
	fi

run-gradio:
	@echo "Launching Gradio dashboard..."
	@chmod +x scripts/run.sh
	./scripts/run.sh

docker-build:
	@echo "Build Docker image (installs CPU PyTorch)."
	docker build -t meeting-transcriber .

clean:
	@echo "Cleaning virtualenv and artifacts"
	rm -rf $(VENV) sample_audio_out.json out.json .chromadb

test:
	@echo "Running test suite..."
	. $(VENV)/bin/activate && python -m pytest tests/ -v
