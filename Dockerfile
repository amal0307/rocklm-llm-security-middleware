FROM python:3.9-slim

WORKDIR /app

# Install system deps for spacy model
RUN apt-get update && \
    apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .
COPY src/api/ ./src/api/
COPY docs/ ./docs/
COPY .github/ ./.github/

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
