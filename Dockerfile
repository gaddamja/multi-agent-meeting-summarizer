FROM python:3.12-slim

# Install ffmpeg and build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install CPU PyTorch and other requirements
RUN pip install --upgrade pip && \
    pip install torch==2.2.0+cpu torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu || true
RUN pip install -r /app/requirements.txt

COPY . /app

ENTRYPOINT ["python", "scripts/run_transcription.py"]
