PYTHON?=python3
VENV=.venv

.PHONY: install-venv fetch-sample transcribe clean docker-build

install-venv:
	@echo "Creating virtualenv at $(VENV) and installing dependencies (except torch)."
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel
	@echo "IMPORTANT: Install PyTorch per https://pytorch.org/get-started/locally/ before installing requirements.txt"
	@echo "Example (macOS CPU): pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
	. $(VENV)/bin/activate && pip install -r requirements.txt

fetch-sample:
	@echo "Fetch sample audio from HF dataset (uses scripts/fetch_hf_audio.py)."
	. $(VENV)/bin/activate && python scripts/fetch_hf_audio.py --repo edinburghcstr/ami --config ihm --match "EN2001a" --output ami_sample.wav || true

transcribe:
	@echo "Run transcription on ami_sample.wav"
	. $(VENV)/bin/activate && python scripts/run_transcription.py ami_sample.wav --model small --output ami_sample_out.json || true

docker-build:
	@echo "Build Docker image (installs CPU PyTorch)."
	docker build -t meeting-transcriber .

clean:
	@echo "Cleaning virtualenv and artifacts"
	rm -rf $(VENV) ami_sample.wav ami_sample_out.json out.json
