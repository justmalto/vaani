FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY ser_masked_attn_best.pt /app/model/ser_masked_attn_best.pt

# Install torch CPU
RUN pip install --no-cache-dir \
    torch==2.9.0+cpu \
    torchaudio==2.9.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
