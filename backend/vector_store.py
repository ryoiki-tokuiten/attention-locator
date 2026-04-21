"""
backend/vector_store.py

Ingestion and search logic for the pgvector-backed neural residual store.

Every row in `sentence_layer_states` holds the mean-pooled residual stream
state for one sentence at one transformer layer, for either the 'residual_input'
(entering the attention block) or 'residual_after_attn' (after the attention
block adds back to the residual stream).

Because these vectors all live in the same hidden_size-dimensional space at
every layer, cosine similarity across layers is geometrically valid.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Maximum vector dimension accepted by the schema (VECTOR(1536)).
# Models with hidden_size > 1536 will need a schema ALTER; in practice
# Gemma-270M has hidden_size=1152, well within this limit.
MAX_SCHEMA_DIM = 1536

VectorType = str  # 'residual_input' | 'residual_after_attn'


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mean_pool(array: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Mean-pool rows [start:end] of `array` along axis 0.
    Returns a 1-D float32 vector of shape (array.shape[-1],).
    If the span is empty, returns a zero vector.
    """
    span = array[start:end]
    if span.shape[0] == 0:
        return np.zeros(array.shape[-1], dtype=np.float32)
    return span.mean(axis=0).astype(np.float32)


def _pad_to_dim(vec: np.ndarray, target_dim: int) -> np.ndarray:
    """Zero-pad a 1-D vector to target_dim if it is shorter."""
    if len(vec) >= target_dim:
        return vec[:target_dim]
    return np.pad(vec, (0, target_dim - len(vec)), mode="constant")


def _to_pgvector(vec: np.ndarray) -> list[float]:
    """Convert a numpy 1-D array to a list of Python floats for asyncpg."""
    padded = _pad_to_dim(vec, MAX_SCHEMA_DIM)
    return padded.tolist()


# ── Ingestion ─────────────────────────────────────────────────────────────────

async def ingest_document(
    pool,
    *,
    markdown_text: str,
    model_name: str,
    sentences: list[dict[str, Any]],
    sentence_boundaries: list[int],
    residual_inputs_path: str,
    residual_after_attention_path: str,
    layer_count: int,
    head_count: int,
    hidden_size: int,
    head_dim: int,
    token_count: int,
) -> int:
    """
    Ingest residual stream vectors for a document into PostgreSQL.

    Parameters
    ----------
    pool:
        asyncpg connection pool (with pgvector codec already registered).
    markdown_text:
        The raw markdown that was analyzed. Used only to compute content_hash.
    model_name:
        String identifier of the model (e.g. 'gemma-3-270m-it-Q8_0.gguf').
    sentences:
        List of sentence payload dicts from the SAVIS pipeline. Each must have
        keys: index, sentence, weight (optional), band (optional).
    sentence_boundaries:
        Token boundary indices from ISA. sentence_boundaries[i] is the start
        token index of sentence i; sentence_boundaries[i+1] is its end.
    residual_inputs_path:
        Absolute path to the residual_inputs.npy file from DeltaTensorStore.
        Shape: [layer_count, token_count, hidden_size].
    residual_after_attention_path:
        Absolute path to the residual_after_attention.npy file.
        Shape: [layer_count, token_count, hidden_size].
    layer_count, head_count, hidden_size, head_dim, token_count:
        Model architecture parameters captured during delta capture.

    Returns
    -------
    int
        The database id of the upserted document row.
    """
    content_hash = compute_content_hash(markdown_text)

    # Load tensors from disk via memmap (read-only; no extra RAM copy).
    ri  = np.load(residual_inputs_path,          mmap_mode="r", allow_pickle=False)
    raa = np.load(residual_after_attention_path, mmap_mode="r", allow_pickle=False)

    if hidden_size > MAX_SCHEMA_DIM:
        raise ValueError(
            f"Model hidden_size {hidden_size} exceeds the schema maximum {MAX_SCHEMA_DIM}. "
            "Run ALTER TABLE sentence_layer_states ALTER COLUMN embedding TYPE vector(<new_dim>)."
        )

    async with pool.acquire() as conn:
        async with conn.transaction():

            # Upsert document row (idempotent via content_hash).
            doc_id: int = await conn.fetchval(
                """
                INSERT INTO documents
                    (content_hash, model_name, token_count, layer_count,
                     head_count, hidden_size, head_dim)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (content_hash) DO UPDATE
                    SET model_name  = EXCLUDED.model_name,
                        token_count = EXCLUDED.token_count
                RETURNING id
                """,
                content_hash, model_name, token_count, layer_count,
                head_count, hidden_size, head_dim,
            )

            inserted_sentences = 0
            inserted_vectors   = 0

            for sent in sentences:
                idx     = int(sent["index"])
                t_start = int(sentence_boundaries[idx])
                t_end   = int(sentence_boundaries[idx + 1]) if idx + 1 < len(sentence_boundaries) else token_count
                t_end   = min(t_end, token_count)

                sent_id: int = await conn.fetchval(
                    """
                    INSERT INTO sentences
                        (document_id, sentence_index, text,
                         token_start, token_end, attention_score, band)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (document_id, sentence_index) DO UPDATE
                        SET text            = EXCLUDED.text,
                            token_start     = EXCLUDED.token_start,
                            token_end       = EXCLUDED.token_end,
                            attention_score = EXCLUDED.attention_score,
                            band            = EXCLUDED.band
                    RETURNING id
                    """,
                    doc_id, idx, sent.get("sentence", ""),
                    t_start, t_end,
                    sent.get("weight"),
                    sent.get("band"),
                )
                inserted_sentences += 1

                # Batch-insert all layer vectors for this sentence.
                records = []
                for layer_idx in range(layer_count):
                    vec_in  = _to_pgvector(_mean_pool(ri[layer_idx],  t_start, t_end))
                    vec_out = _to_pgvector(_mean_pool(raa[layer_idx], t_start, t_end))

                    records.append((sent_id, layer_idx, "residual_input",      vec_in))
                    records.append((sent_id, layer_idx, "residual_after_attn", vec_out))

                await conn.executemany(
                    """
                    INSERT INTO sentence_layer_states
                        (sentence_id, layer_index, vector_type, embedding)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (sentence_id, layer_index, vector_type) DO NOTHING
                    """,
                    records,
                )
                inserted_vectors += len(records)

            logger.info(
                "Ingested document %s: %d sentences, %d layer-state vectors.",
                content_hash[:12], inserted_sentences, inserted_vectors,
            )

    return doc_id


# ── Search ────────────────────────────────────────────────────────────────────

async def search_by_layer_vector(
    pool,
    *,
    query_vector: np.ndarray,
    target_layer: int,
    vector_type: VectorType = "residual_after_attn",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Find the `limit` nearest sentences (by cosine distance) whose residual
    state at `target_layer` is closest to `query_vector`.

    Parameters
    ----------
    pool:
        asyncpg pool with pgvector codec registered.
    query_vector:
        1-D numpy array of shape (hidden_size,). Will be padded to MAX_SCHEMA_DIM.
    target_layer:
        The transformer layer to search in.
    vector_type:
        'residual_input' or 'residual_after_attn'.
    limit:
        Number of results to return.

    Returns
    -------
    List of dicts with keys:
        sentence_id, sentence_text, sentence_index, attention_score, band,
        document_id, content_hash, model_name, layer_index, cosine_distance.
    """
    qvec = _to_pgvector(query_vector.astype(np.float32))

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                s.id                               AS sentence_id,
                s.text                             AS sentence_text,
                s.sentence_index,
                s.attention_score,
                s.band,
                d.id                               AS document_id,
                d.content_hash,
                d.model_name,
                sls.layer_index,
                sls.embedding <=> $1               AS cosine_distance
            FROM sentence_layer_states sls
            JOIN sentences  s ON sls.sentence_id = s.id
            JOIN documents  d ON s.document_id   = d.id
            WHERE sls.vector_type  = $2
              AND sls.layer_index  = $3
            ORDER BY cosine_distance ASC
            LIMIT $4
            """,
            qvec, vector_type, target_layer, limit,
        )

    return [dict(row) for row in rows]


async def get_cross_layer_drift(
    pool,
    *,
    sentence_id: int,
) -> list[dict[str, Any]]:
    """
    For a given sentence, return the pairwise cosine similarity between every
    pair of adjacent layers using the 'residual_after_attn' vector type.

    This exposes at which transformer layer the model's internal representation
    of the sentence changed the most (lowest cosine similarity = highest drift).

    Returns
    -------
    List of dicts with keys: from_layer, to_layer, cosine_similarity.
    Ordered by cosine_similarity ASC (most drifted pair first).
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                a.layer_index                         AS from_layer,
                b.layer_index                         AS to_layer,
                1 - (a.embedding <=> b.embedding)     AS cosine_similarity
            FROM sentence_layer_states a
            JOIN sentence_layer_states b
                ON  a.sentence_id = b.sentence_id
                AND b.layer_index = a.layer_index + 1
            WHERE a.sentence_id = $1
              AND a.vector_type  = 'residual_after_attn'
              AND b.vector_type  = 'residual_after_attn'
            ORDER BY cosine_similarity ASC
            """,
            sentence_id,
        )

    return [dict(row) for row in rows]


async def get_document_layer_states(
    pool,
    *,
    document_id: int,
    vector_type: VectorType = "residual_after_attn",
) -> list[dict[str, Any]]:
    """
    Retrieve all sentence-layer state metadata (without raw embeddings)
    for a given document. Used by the frontend to populate layer selectors.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                s.id            AS sentence_id,
                s.sentence_index,
                s.text,
                s.attention_score,
                s.band,
                sls.layer_index
            FROM sentence_layer_states sls
            JOIN sentences s ON sls.sentence_id = s.id
            WHERE s.document_id = $1
              AND sls.vector_type = $2
            ORDER BY s.sentence_index, sls.layer_index
            """,
            document_id, vector_type,
        )

    return [dict(row) for row in rows]
