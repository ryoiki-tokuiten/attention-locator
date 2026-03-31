# Attention Markdown

In practice, this lets you explore how an LLM appears to "see" a piece of text by turning inter-sentence attention into an interactive reading experience instead of only showing raw matrices.


## What This App Does

Given markdown input, the application:

1. extracts readable sentence content from the document
2. computes sentence-level attention relationships with a local model
3. normalizes those scores into emphasis bands
4. lets you inspect both sentence-level and token-level attention views in the UI
5. feeds the results into a custom markdown renderer so the most semantically attended text stands out visually

This makes the project useful both as:

- an implementation of the original SAVIS idea
- an interactive prototype for attention-based text visualization

## How To Start The Application

### 1. Start the backend

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Use `backend/requirements-cpu.txt` instead if you explicitly want CPU-only PyTorch wheels.

If you place a local `.gguf` model inside [`model/`](./model), the backend will auto-discover it on startup. With the current files in this repo, it will prefer the smaller Gemma checkpoint over the larger Gemma and unsupported Qwen GGUFs, which keeps the app responsive on typical laptop GPUs.

The backend starts on `http://localhost:8000`.
The model is lazy-loaded on the first request, so the first `/health`, `/analyze`, or attention request can take a few seconds.

### 2. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on Vite's local dev server, usually `http://localhost:5173`.

By default it calls the backend at `http://localhost:8000`. If you need a different backend URL, set `VITE_API_BASE_URL`.

The UI includes:

- the rendered markdown view
- a sentence attention matrix view
- a token attention drill-down view for a selected sentence pair


## Core Components

- `backend/analysis.py`: converts markdown into analyzable text, scores sentences, and prepares response payloads for the UI
- `backend/savis/attention.py`: computes inter-sentence attention from model outputs
- `backend/savis/generator.py`: handles text generation and attention extraction
- `backend/savis/visualization.py`: provides attention visualization utilities
- `frontend/RenderMathMarkdown.tsx` and `frontend/RenderMathMarkdownLogic.ts`: render markdown with your custom attention-aware emphasis layer
