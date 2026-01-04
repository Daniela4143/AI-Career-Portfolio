# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 直接安裝到系統目錄，下一階段再整個拷貝，這樣路徑最乾淨
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim

WORKDIR /app

# 直接從 builder 的 python site-packages 拷貝到 runtime
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 複製程式與數據
COPY .env.example .env 
COPY app.py app.py
COPY data/ ./data/

ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

EXPOSE 8080

CMD ["streamlit", "run", "app.py"]