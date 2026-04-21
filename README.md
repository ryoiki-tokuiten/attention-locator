# Attention Markdown

In practice, this lets you explore how an LLM appears to "see" a piece of text by turning inter-sentence attention into an interactive reading experience instead of only showing raw matrices.


## What This App Does

Given markdown input, the application:

1.  **Extracts** readable sentence content from the document.
2.  **Computes** sentence-level attention relationships with a local model.
3.  **Normalizes** those scores into emphasis bands.
4.  **Inspects** both sentence-level and token-level attention views in the UI.
5.  **Renders** the results into a custom markdown viewer where semantically attended text stands out visually.
6.  **(New) Persists** neural residual states to a PostgreSQL database via `pgvector` for cross-document semantic search and representational drift analysis.

This makes the project useful both as:

- An implementation of the original SAVIS idea.
- An interactive prototype for attention-based text visualization.
- A **Design and Analysis of Algorithms (DAA)** project (complex attention pooling and opportunity-corrected scoring).
- A **Database Management Systems (DBMS)** project (high-dimensional indexing with pgvector and complex relational joins).


## Neural Residual Vector Store

The application includes a persistent **Neural Vector Registry** that allows you to:

*   **Ingest Documents:** Store the mean-pooled residual stream states (hidden states) for every sentence at every layer of the model.
*   **ANN Semantic Search:** Find sentences across all analyzed documents whose neural representation at a specific layer is closest to a query sentence using HNSW indexes.
*   **Layer Drift Analysis:** Visualize how a sentence's representation transforms as it moves through the transformer layers via self-join similarity queries.


## How To Start The Application

### 1. Database Setup (Optional)

The application uses **PostgreSQL 17** with the **pgvector** extension for persistent storage.

1.  **Create the database:**
    ```bash
    psql -U postgres -c "CREATE DATABASE attention_markdown;"
    ```
2.  **Run the schema migration:**
    ```bash
    # Default password is 'roshan'. Update accordingly.
    PGPASSWORD=roshan psql -U postgres -d attention_markdown -f backend/schema.sql
    ```

### 2. Start the backend

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt

# Set the DB password if different from the default
export ATTN_DB_PASSWORD=roshan 

uvicorn backend.main:app --reload
```

Use `backend/requirements-cpu.txt` instead if you explicitly want CPU-only PyTorch wheels.

If you place a local `.gguf` model inside [`model/`](./model), the backend will auto-discover it on startup. 

#### Database Environment Variables
| Variable | Default | Description |
|---|---|---|
| `ATTN_DB_HOST` | `localhost` | PostgreSQL host |
| `ATTN_DB_PORT` | `5432` | PostgreSQL port |
| `ATTN_DB_NAME` | `attention_markdown` | Database name |
| `ATTN_DB_USER` | `postgres` | Database user |
| `ATTN_DB_PASSWORD` | `roshan` | Database password |

### 3. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend runs on Vite's local dev server, usually `http://localhost:5173`.


## Core Components

- `backend/analysis.py`: Converts markdown into analyzable text and scores sentences.
- `backend/vector_store.py`: **(New)** Ingestion and search logic for the pgvector-backed neural residual store.
- `backend/db.py`: **(New)** Database connection pool management with pgvector codec registration.
- `backend/schema.sql`: **(New)** PostgreSQL schema and HNSW index definitions.
- `backend/savis/attention.py`: Computes inter-sentence attention from model outputs.
- `backend/savis/generator.py`: Handles text generation and attention extraction.
- `frontend/RenderMathMarkdown.tsx` and `frontend/RenderMathMarkdownLogic.ts`: Custom markdown rendering with emphasis layers.
