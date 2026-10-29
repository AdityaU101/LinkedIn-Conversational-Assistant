FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt \
    && python -m spacy download en_core_web_sm
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# Data/ is mounted as a volume (see docker-compose.yml); build indexes on the host or via
# POST /api/admin/ingestion/run before first use.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
