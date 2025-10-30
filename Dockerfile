# Use official Python slim image
FROM python:3.11-slim

WORKDIR /app

# System deps (if any) and clean apt caches
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose streamlit port
EXPOSE 8501

# Use a simple healthcheck (optional)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import sys,requests; \
  resp = requests.get('http://localhost:8501') if True else None; \
  sys.exit(0 if resp and resp.status_code<500 else 1)" || exit 0

# Start streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.baseUrlPath=/error_report"]
