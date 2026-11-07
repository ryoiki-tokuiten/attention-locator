-- Attention-Locator — Neural Residual Vector Store

CREATE EXTENSION IF NOT EXISTS vector;

-- A document that was analyzed.
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    content_hash    TEXT UNIQUE NOT NULL,
    model_name      TEXT NOT NULL,
    token_count     INT  NOT NULL,
    layer_count     INT  NOT NULL,
    head_count      INT  NOT NULL,
    hidden_size     INT  NOT NULL,
    head_dim        INT  NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- A single sentence extracted from a document.
CREATE TABLE IF NOT EXISTS sentences (
    id              SERIAL PRIMARY KEY,
    document_id     INT  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    sentence_index  INT  NOT NULL,
    text            TEXT NOT NULL,
    token_start     INT  NOT NULL,
    token_end       INT  NOT NULL,
    attention_score FLOAT,
    band            TEXT CHECK (band IN ('low', 'medium', 'high')),
    UNIQUE (document_id, sentence_index)
);

-- Core vector table.
-- One row per (sentence, layer, vector_type).
-- vector_type is one of:
--   'residual_input'      — hidden state entering the attention block at this layer
--   'residual_after_attn' — hidden state after attention + residual add
-- The embedding dimension matches hidden_size (1152 for Gemma-270M).
-- We store as VECTOR(1536) to accommodate models up to that size;
-- actual vectors are zero-padded or stored at their native dimension via a
-- runtime check. For simplicity we pin to 1536 and cast on insert.
CREATE TABLE IF NOT EXISTS sentence_layer_states (
    id              SERIAL PRIMARY KEY,
    sentence_id     INT  NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    layer_index     INT  NOT NULL,
    vector_type     TEXT NOT NULL CHECK (vector_type IN ('residual_input', 'residual_after_attn')),
    embedding       VECTOR(1536),
    UNIQUE (sentence_id, layer_index, vector_type)
);

-- HNSW index for approximate nearest-neighbour search using cosine distance.
-- m=16 and ef_construction=64 are the pgvector defaults; good for most workloads.
CREATE INDEX IF NOT EXISTS idx_sentence_layer_states_hnsw
    ON sentence_layer_states
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Covering index to speed up layer+type filtered queries.
CREATE INDEX IF NOT EXISTS idx_sentence_layer_states_layer_type
    ON sentence_layer_states (layer_index, vector_type);

-- Index to speed up joins from sentence_layer_states → sentences → documents.
CREATE INDEX IF NOT EXISTS idx_sentences_document_id
    ON sentences (document_id);
