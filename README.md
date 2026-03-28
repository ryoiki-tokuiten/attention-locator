# SAVIS Attention Markdown

The project combines:
- the SAVIS sentence-level attention pipeline
- a FastAPI backend that computes sentence salience from model attention
- a React/Vite frontend that renders markdown with your custom attention visualization treatment

In practice, this lets you explore how an LLM appears to "see" a piece of text by turning inter-sentence attention into an interactive reading experience instead of only showing raw matrices.

This work has been accepted to NAACL 2025 (Annual Conference of the North American Chapter of the Association for Computational Linguistics) Demo. [Read the paper](https://aclanthology.org/2025.naacl-demo.27). Citation details are available in the [Citation](#citation) section.

<center>
<img src="https://raw.githubusercontent.com/Seongbuming/savis/master/images/savis.png" width="500" alt="SAVIS analyzing reviews">
</center>

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

## Project Structure

```text
backend/
  main.py            FastAPI app
  analysis.py        markdown preprocessing + sentence weighting pipeline
  savis/             SAVIS implementation moved here from src/
  tests/             backend test suite

frontend/
  src/               React app
  RenderMathMarkdown.tsx
  RenderMathMarkdownLogic.ts
```

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

The UI now includes:

- the rendered markdown view
- a sentence attention matrix view
- a token attention drill-down view for a selected sentence pair

## SAVIS Package Usage

The original SAVIS implementation now lives in [`backend/savis`](./backend/savis).

If you want to use it as a Python package from this repository, install the project in editable mode:

```bash
pip install -e .
```

Then:

```python
from savis import TextGenerator, ISA, ISAVisualization

generator = TextGenerator("Model name")
input_text = "Your input prompt here"
generated_text, attentions, tokenizer, input_ids, outputs = generator.generate_text(input_text)

isa = ISA(outputs.sequences[0], attentions, tokenizer)
vis = ISAVisualization(isa.sentence_attention, isa.sentences)
vis.visualize_sentence_attention()
```

If you are importing directly from the source tree without installing the package, use:

```python
from backend.savis import TextGenerator, ISA, ISAVisualization
```

## Core Components

- `backend/analysis.py`: converts markdown into analyzable text, scores sentences, and prepares response payloads for the UI
- `backend/savis/attention.py`: computes inter-sentence attention from model outputs
- `backend/savis/generator.py`: handles text generation and attention extraction
- `backend/savis/visualization.py`: provides attention visualization utilities
- `frontend/RenderMathMarkdown.tsx` and `frontend/RenderMathMarkdownLogic.ts`: render markdown with your custom attention-aware emphasis layer
