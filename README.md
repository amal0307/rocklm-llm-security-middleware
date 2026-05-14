# RockLM — Modular AI Security Middleware

**A pluggable, phase-driven security middleware framework for LLMs, RAG systems, and AI agents.**

RockLM intercepts every prompt and response flowing through your AI pipeline, applying multi-layered security checks — from input sanitization and prompt injection detection to PII filtering and output validation — before anything reaches the model or the user.

---

## Benchmark Results

Evaluated on the **[deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections)** dataset (546 samples):

| Metric | Value |
|---|---|
| **Accuracy** | 99.63% |
| **Precision** | 99.51% |
| **Recall** | 99.51% |
| **F1 Score** | 0.9951 |
| **False Positive Rate** | 0.29% |
| **False Negative Rate** | 0.49% |
| **Avg Latency** | 0.246s/prompt |

> 202/203 injection attacks blocked · 342/343 benign prompts allowed through · Only 1 false positive

---

## Features

### Security Modules

| Module | Description |
|---|---|
| **InputSanitizer** | Multi-layer input validation with Unicode normalization, pattern-based injection scoring, Mahalanobis anomaly detection, and rate limiting |
| **AgentContextTracker** | Tracks conversation context drift using semantic embeddings (`all-MiniLM-L6-v2`) to detect manipulation attempts across turns |
| **AgentPermissionEnforcer** | Role-based access control with configurable permission policies and risk-level scoring |
| **OutputFilter** | PII detection & redaction (email, phone, SSN, credit card), toxicity classification via `toxic-bert`, and content policy enforcement |
| **IntegrityChecker** | Validates data integrity across the pipeline to prevent tampering |
| **RetrievalMonitor** | Monitors RAG retrieval patterns for anomalous behavior |

### Architecture

- **Plugin Architecture** — Modular design via `RockLMProtocol`; add/remove security modules without touching core code
- **Phase-Driven Pipeline** — Sequential processing: Input → Sanitize → Track → Enforce → Generate → Filter → Output
- **Real-Time Inference** — Gradio web interface for interactive prompt testing
- **REST API** — FastAPI endpoint for production integration
- **Docker Ready** — Dockerfile included for containerized deployment

---

## Project Structure

```
rocklm/
├── app.py                          # Gradio web interface
├── main.py                         # CLI entry point
├── benchmark.py                    # Benchmark script (deepset/prompt-injections)
├── requirements.txt
├── Dockerfile
├── src/
│   ├── core/
│   │   ├── config.py               # SecurityConfig with env-based overrides
│   │   ├── pipeline.py             # Main security pipeline orchestrator
│   │   ├── protocol.py             # RockLMModule base class & plugin interface
│   │   ├── plugin_manager.py       # Dynamic module loader
│   │   └── logger.py               # Structured logging
│   ├── modules/
│   │   ├── input_sanitizer.py      # Input validation & injection detection
│   │   ├── agent_tracker.py        # Context drift tracking (MiniLM embeddings)
│   │   ├── agent_permission_enforcer.py  # RBAC & permission enforcement
│   │   ├── output_filter.py        # PII redaction & toxicity filtering
│   │   ├── integrity_checker.py    # Data integrity validation
│   │   └── retrieval_monitor.py    # RAG retrieval monitoring
│   ├── api/
│   │   └── app.py                  # FastAPI REST API
│   └── utils/
│       └── helpers.py
├── tests/                          # Unit tests for all modules
├── docs/
│   └── architecture.md
└── logs/                           # Benchmark results (gitignored)
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/amal0307/rocklm-llm-security-middleware.git
cd rocklm-llm-security-middleware
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
LLM_MODEL_NAME=gemini-2.0-flash
EMBEDDING_MODEL=models/text-embedding-004
```

### 3. Run the Gradio Interface

```bash
python app.py
```

Open `http://127.0.0.1:7861` in your browser.

### 4. Run the REST API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 5. Run the Benchmark

```bash
python benchmark.py
```

Results are saved to `logs/benchmark_results.json` and `logs/benchmark_results.log`.

---

## How It Works

```
User Prompt
    │
    ▼
┌─────────────────────┐
│   InputSanitizer     │  ← Unicode normalization, injection pattern scoring
│   (Priority 1)       │     Rate limiting, Mahalanobis anomaly detection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ AgentContextTracker  │  ← Semantic embedding drift detection
│   (Priority 2)       │     Tracks context manipulation across turns
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ PermissionEnforcer   │  ← Role-based access control
│   (Priority 3)       │     Risk-level scoring
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     LLM Call         │  ← Gemini API (or any configured LLM)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   OutputFilter       │  ← PII redaction, toxicity detection
│   (Priority 5)       │     Content policy enforcement
└────────┬────────────┘
         │
         ▼
    Safe Response
```

---

## Tech Stack

- **Python 3.10+**
- **Gradio** — Interactive web interface
- **FastAPI** — REST API
- **Transformers** — `toxic-bert` (toxicity), `all-MiniLM-L6-v2` (embeddings)
- **spaCy** — Named entity recognition for PII detection
- **Google Gemini** — LLM backend
- **PyTorch** — Model inference
- **Docker** — Containerized deployment

---

## License

MIT License

---

## Author

**Amal Raju** — [GitHub](https://github.com/amal0307)