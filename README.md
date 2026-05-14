# RockLM: Modular AI Security Framework

RockLM is a pluggable, phase‑driven AI security middleware for LLMs, RAG systems, and agents.

## Features

**1.**Input sanitization, prompt validation, output filtering, PII detection, rate limiting  
**2.**Agent context tracking & permission enforcement  
**3.**RAG defenses (vector poisoning, retrieval monitoring, integrity checks, activation poisoning)  
**4.**RockLM Protocol & plugin registry for community‑driven extensions  
**5.**REST API (FastAPI) and Gradio demo (Hugging Face Spaces)

## Quickstart

```bash
git clone <your‑repo>
cd ai‑security‑pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000