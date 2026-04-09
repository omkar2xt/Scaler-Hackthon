FROM python:3.11-slim@sha256:8119cc7f21b9d2b850ae5ad65d7e93a73fb907e04c6f91a79fa1fbf88aaac46d

WORKDIR /app

# Copy requirements first for better caching
COPY misinfoguard_env/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Health check using Python to verify the app is running
# For FastAPI app: run uvicorn on Spaces default port 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health').read(); exit(0)" || exit 1

# Run the API server for Spaces/OpenEnv connectivity
CMD ["uvicorn", "misinfoguard_env.app:app", "--host", "0.0.0.0", "--port", "7860"]
