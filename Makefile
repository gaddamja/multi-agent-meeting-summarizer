PYTHON?=python3
VENV=.venv

.PHONY: install-venv transcribe clean docker-build

install-venv:
	@echo "Creating virtualenv at $(VENV) and installing dependencies (except torch)."
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel
	@echo "IMPORTANT: Install PyTorch per https://pytorch.org/get-started/locally/ before installing requirements.txt"
	@echo "Example (macOS CPU): pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
	. $(VENV)/bin/activate && pip install -r requirements.txt

transcribe:
	@echo "Run transcription on the included sample audio file"
	. $(VENV)/bin/activate && python scripts/run_transcription.py sample_audio.mp3 --model small --output sample_audio_out.json || true

docker-build:
	@echo "Build Docker image (installs CPU PyTorch)."
	docker build -t meeting-transcriber .

clean:
	@echo "Cleaning virtualenv and artifacts"
	rm -rf $(VENV) sample_audio_out.json out.json
