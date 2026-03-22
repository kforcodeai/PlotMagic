# Streamlit UI

This project includes a rich Streamlit UI for running and inspecting the agentic RAG pipeline.

## Run

```bash
source .env
.venv/bin/streamlit run streamlit_app.py
```

## Single Script (API + UI)

Use one command to run both FastAPI and Streamlit:

```bash
.venv/bin/python scripts/run_dev_stack.py --api-reload
```

Optional custom ports:

```bash
.venv/bin/python scripts/run_dev_stack.py --api-port 8001 --ui-port 8765 --api-reload
```

Stop both with `Ctrl+C`.

## Run FastAPI too

Start API and UI in separate terminals.

Terminal 1:

```bash
source .env
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2:

```bash
source .env
.venv/bin/streamlit run streamlit_app.py --server.port 8501
```

Optional single-command background run:

```bash
source .env
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
.venv/bin/streamlit run streamlit_app.py --server.port 8501
```

If port `8501` is already in use, choose another port (for example `8765`) or stop the existing process:

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
kill <PID>
```

## What it shows

- Final compliance brief (verdict, conditions, actions, risk flags)
- Citation explorer with API deep links (`/rules/{document_id}/source#{anchor}`)
- Provider-specific rendering:
  - `no_llm`: emphasizes retrieved chunks + citation traceability for retrieval verification.
  - `openai_responses_llm`: renders the final brief with inline citation links per supported claim.
- Thinking/tools panel:
  - high-level agent trace
  - live tool-stage events (scope, occupancy, retrieval passes, evidence judge, llm provider)
  - grounding payload and latency map
- Raw JSON output for verification/debugging

## Perceived-latency behavior

- The UI streams live pipeline events while the backend query runs in a worker thread.
- Final summary text is streamed token-by-token to reduce perceived wait time.
