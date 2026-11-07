# Attention-Locator Pipeline Flow

## What Text the Model Sees

The model sees the raw document text as-is. We only normalize line endings (`\r\n` and `\r` to `\n`) so character indexing stays consistent across platforms. Nothing is stripped or pre-cleaned before inference: every word, punctuation mark, symbol, and newline goes directly into the input string.

---

## Who Tokenizes the Text

The model's own tokenizer does.

The sentence detector does not decide what a token is. The Hugging Face tokenizer belonging to the loaded language model takes the raw text string and breaks it into model-native subword tokens. Depending on the tokenizer, a single word may become:
- one token
- several subword tokens
- tokens with leading-space markers (such as `Ġ` or `▁`)
- punctuation-only tokens
- newline tokens

This token sequence is the true atomic unit that the transformer's attention mechanism operates on.

---

## Single Full-Context Pass (No Chunking)

The entire document is sent through the model in one forward pass, as long as it fits within `max_input_tokens`.

Conceptually, the sequence is:

`raw text -> model tokenizer -> full token sequence -> one forward pass`

There is no sentence batching, no sliding window, and no overlap merging. Every token in the document attends over the exact preceding context it would see during normal model execution.

---

## What Attention Tensor Comes Back from the Model

We run the forward pass with `output_attentions=True` (and `use_cache=False` for causal models) to collect attention weights from every decoder layer.

If:
- `L` = number of transformer layers
- `H` = number of attention heads per layer
- `T` = total token count

then the model returns attention tensors shaped like:

`L x H x T x T`

For each layer `l` and head `h`, you get a 2D matrix:

`A[l, h, i, j]`

where:
- `i` is the query token position (the token doing the looking)
- `j` is the key token position (the token being looked at)
- the value `A[l, h, i, j]` is how much token `i` attends to token `j` inside that specific head and layer

Because this is a causal decoder model, attention is masked to the past: token `i` can only attend to tokens `j <= i`, never to future tokens `j > i`. That makes each `T x T` matrix lower-triangular.

This is the self-attention structure of the document pass itself — how later parts of the text route information from earlier parts of the text.

---

## How Sentences Are Detected

Tokens are defined by the model. Sentences are defined by human text boundaries.

To bridge the two, the backend uses a sentence tokenizer (`nltk.sent_tokenize` with `PunktSentenceTokenizer` as an offline fallback). That gives two separate segmentations of the same document:
- **Token segmentation**: subword pieces produced by the model tokenizer
- **Sentence segmentation**: sentence strings produced by sentence tokenization over decoded text

The pipeline then aligns those two segmentations so every sentence maps to an exact range of model tokens.

---

## How Sentences Are Aligned to Token Spans

In `backend/attention.py` (`ISA._find_sentence_boundaries`), the analyzer walks through the token stream and stops at newline-bearing tokens or the end of the sequence. At each checkpoint, it:

1. Decodes the token sequence seen so far into text.
2. Slices out the newly added text chunk since the last sentence boundary.
3. Runs sentence tokenization on that chunk to get candidate sentences.
4. Advances token-by-token (`idx_j`) and decodes `sequences[slice_start : idx_j + 1]` until the decoded slice contains the current sentence string.
5. Records the token index `idx_j + 1` where that sentence ends in `sentence_boundaries_ids`.

As a result, each sentence `k` becomes an exact token span:

`sentence_k = tokens[b_k : b_{k+1}]`

where `b_k = sentence_boundaries_ids[k]` and `b_{k+1} = sentence_boundaries_ids[k + 1]`.

---

## How Token Attention Becomes Sentence Attention

Once we have:
- raw token attention matrices `A[l, h, i, j]` across all layers and heads
- token spans `[b_k, b_{k+1})` for every sentence

the directional sentence-to-sentence matrix `S` is built in three max-pooling reductions.

For a sentence pair `(p, q)` where `p` is the querying sentence and `q` is the attended-to sentence, the analyzer looks at the rectangle of token pairs where query token `i` belongs to sentence `p` and key token `j` belongs to sentence `q`:

1. **Max over layers:**
   `A_head[h, i, j] = max_l A[l, h, i, j]`

2. **Max over token pairs inside the sentence rectangle `(p, q)` for each head:**
   `S_head[h, p, q] = max_{i in p, j in q} A_head[h, i, j]`

3. **Max over heads:**
   `S[p, q] = max_h S_head[h, p, q]`

Putting all three steps together, each entry in the sentence matrix is:

`S[p, q] = max_{h, l, i in p, j in q} A[l, h, i, j]`

### Why Max-Pooling Instead of Averaging?

This does **not** measure average attention mass across all tokens and heads. In a transformer, most heads in most layers are inactive for any given semantic link, while a single specialized head in a specific layer may fire strongly between two key tokens (for example, a pronoun or summary phrase in sentence `p` attending back to a definition in sentence `q`). Averaging across all `L x H` heads and all token pairs washes out those targeted connections; max-pooling asks:

*"What is the single strongest token-to-token attention link between sentence `p` and sentence `q` across any layer and any head?"*

---

## What the Sentence Matrix Means

In `S[p, q]`:
- **Row `p`** is the querying sentence
- **Column `q`** is the attended-to sentence
- **Value `S[p, q]`** is how strongly sentence `p` attends to sentence `q`

Because the model is causal:
- later sentences (`p > q`) can look back at earlier sentences (`q`)
- earlier sentences (`p < q`) cannot look forward to later sentences

So the matrix is directional: information flows from earlier sentences (`q`) into later sentences (`p`).

---

## How One Sentence Gets Its Final Emphasis Score (Opportunity Correction)

The colored sentence emphasis in **Semantic Surface** is derived from the sentence matrix `S`, with a correction for causal position bias.

### Step A: Incoming Sentence Citations

For sentence `j`, we first sum all attention it receives from other sentences in the document (summing column `j` and subtracting self-attention on the diagonal):

`I_j = sum_i S[i, j] - S[j, j]`

### Step B: Why Raw Incoming Sums Are Biased

If we stopped at `I_j`, sentences near the beginning of the document would almost always win simply because every later sentence in the document is allowed to attend to them, while sentences near the end have very few tokens after them that can look back at them.

### Step C: Later-Token Opportunity Count

To remove this front-loaded bias, `compute_opportunity_corrected_sentence_scores` in `backend/analysis.py` counts how many later tokens even had the *opportunity* to attend to sentence `j`.

Let:
- `b_k` = token boundary index where sentence `k` starts (`b_{k+1}` where it ends)
- `n` = number of usable sentences
- `T = b_n` = total usable token count in the analyzed document

Then the number of tokens appearing after sentence `j` ends is:

`O_j = max(T - b_{j+1}, 1)`

We clamp `O_j` to at least `1` so the denominator for the final sentence is never zero. `O_j` is literally: *"How many later tokens were even allowed to look back at sentence `j`?"*

### Step D: Rescaling by Document-Average Opportunity

Next, we compute the average opportunity count across all sentences in the document:

`O_bar = max((1 / n) * sum_j O_j, 1)`

If we only divided `I_j` by `O_j`, the resulting numbers would shrink to tiny fractions on long documents. Multiplying by `O_bar / O_j` keeps the scores in a natural scale while correcting for position:

`R_j = I_j * (O_bar / O_j)`

Equivalently:

`R_j = (sum_i S[i, j] - S[j, j]) * (O_bar / O_j)`

This does three things at once:
- Sentences near the beginning with *more than average* opportunity get scaled down.
- Sentences near the end with *less than average* opportunity get scaled up.
- The overall scale across the document is preserved.

If sentence-boundary token indices are unavailable or truncated, the scorer falls back to the uncorrected incoming sum `R_j = I_j`.

---

## How Scores Are Normalized and Banded

### Min-Max Normalization

The opportunity-corrected raw scores `R_j` are normalized relative to the current document into `[0, 1]`:

- If all `R_j` values in the document are identical, every sentence gets the neutral midpoint:
  `W_j = 0.5`
- Otherwise:
  `W_j = (R_j - min_k R_k) / (max_k R_k - min_k R_k)`

So a normalized weight `W_j` near `1.0` means:
- this sentence received the strongest incoming attention in the current document *after* accounting for how many later tokens had the chance to attend to it.

It does **not** mean the sentence has "100% absolute importance" outside this document.

### Low / Medium / High Score Bands

After normalization, `assign_score_bands` in `backend/analysis.py` assigns each sentence to a `low`, `medium`, or `high` band using document-relative percentiles:
- If the spread `max(W) - min(W)` is very flat (`< 0.08`), all sentences default to `medium`.
- Otherwise, the `high` cutoff is computed from the `78th` percentile (clamped to `[0.48, 0.82]`) and the `medium` cutoff from the `38th` percentile (clamped to `[0.18, high_cutoff - 0.08]`).

---

## How Source Character Offsets Are Recovered

Once the backend has the list of sentence strings, `map_sentences_to_source_offsets` in `backend/attention.py` maps each sentence back onto character indices in the raw document text:

1. It builds a normalized version of the source text (collapsing whitespace runs to a single space, lowercasing characters, and stripping zero-width characters) alongside an `offset_map` array that records the original character index in the raw string for every normalized character.
2. For each sentence, it normalizes the sentence string the same way and searches forward from `search_start` in the normalized source string.
3. Using `offset_map`, it converts the match start and end indices back into exact `[sourceStart, sourceEnd]` character offsets in the original raw text.

These `[sourceStart, sourceEnd]` offsets let the frontend wrap each sentence directly in the plain text surface without guessing sentence boundaries in the browser.

---

## Residual Stream Delta Capture (How `Embeddings Delta` Works)

Raw attention weights `A[l, h, i, j]` tell you where a head looked, but they do not tell you how much that head actually changed the residual stream. A head can have high attention weight on a token whose value vector `V` has tiny norm or projects to near-zero through `o_proj`.

To measure what attention blocks actually **write** into the residual stream, `capture_delta_document` in `backend/analysis.py` attaches PyTorch forward hooks to every decoder layer during a single forward pass.

### What the Hooks Intercept at Every Layer `l`

1. **Layer Pre-Hook (`residual_input`)**:
   Captures the hidden state tensor `residual_input` of shape `[T, hidden_size]` entering decoder layer `l` before layer norm and self-attention.

2. **Self-Attention Hook**:
   - Captures `attn_output` (`[T, hidden_size]`) and `attn_weights` (`[H, T, T]`).
   - Recomputes the per-head value vectors `value_states = v_proj(normalized_hidden_states)` and repeats KV heads (`repeat_key_value_heads`) if the model uses Grouped-Query Attention (GQA), yielding shape `[H, T, head_dim]`.
   - Computes the per-head context output before output projection:
     `head_context = matmul(attn_weights, value_states)` (`[T, H, head_dim]`).
   - Reshapes the output projection matrix `o_proj.weight` into per-head blocks of shape `[H, hidden_size, head_dim]`.
   - Uses `torch.einsum` to project both the individual token value states and the mixed head contexts into the full `hidden_size` residual space:
     - `projected_value_states`: `einsum("bhtd,hod->bhto", value_states, projection_blocks)` (`[H, T, hidden_size]`) — what token `t` offers to write through head `h` into the residual stream.
     - `head_projected_delta`: `einsum("bthd,hod->btho", head_context, projection_blocks)` (`[T, H, hidden_size]`) — the actual vector written by head `h` at token `t`.
   - Applies the layer's `post_attention_layernorm` (if present) to get `attention_delta_post_norm` (`[T, hidden_size]`), which is the exact delta added to the residual stream:
     `residual_after_attention = residual_input + attention_delta_post_norm`

All captured tensors are written layer-by-layer to memory-mapped `.npy` files on disk via `DeltaTensorStore` (`backend/delta_capture.py`) so GPU memory is freed immediately after the forward pass.

### How Sentence-Level Delta Metrics Are Derived (`backend/delta_visualization.py`)

From the stored `.npy` tensors, `build_delta_visualization_payload` computes how much each source token `j` writes into each target token `i` across layers and heads, and aggregates those writes by sentence spans `[b_k, b_{k+1})`:

- **Incoming Flow (`incomingFlow`, `incomingFlowNorm`)**:
  How strongly earlier context writes into sentence `p` across layers, normalized to `[0, 1]`.
- **Context Distance Mix (`localShare`, `mediumShare`, `globalShare`)**:
  Splits the incoming writes into sentence `p` by how far back the source sentence `q` is:
  - **Local**: adjacent / nearby sentences
  - **Medium**: mid-range sentences
  - **Global**: distant earlier sentences
- **Source Composition (`topSources`, `globalTopEdges`)**:
  Ranks the specific earlier sentences `q < p` that contribute the largest residual stream writes into sentence `p`. This powers both the `Source composition` mode (showing top upstream contributors for a clicked sentence) and `Flow map` mode (drawing the strongest sentence-to-sentence residual write paths across the document).
- **Layer Drift (`drift`, `driftNorm`, `peakLayer`, `driftByLayer`)**:
  Measures how much a sentence's representation changes across layers (comparing `residual_input` and `residual_after_attention`), identifying the `peakLayer` where the sentence undergoes its largest update.

---

## What Each Frontend View Shows

### Pinned Surface Modes (`Semantic Surface` & `Embeddings Delta`)

In `frontend/render.tsx`, the plain text is sliced using the backend's `[sourceStart, sourceEnd]` character offsets:
- In **Semantic Surface**, each sentence range is wrapped in `<span class="semantic-sentence">` colored by its normalized opportunity-corrected weight `W_j` and band (`low`, `medium`, `high`).
- In **Embeddings Delta**, each sentence range is wrapped in `<span class="delta-token">` styled according to the active delta sub-mode (`Local vs global`, `Flow map`, `Source composition`, or `Layer drift`).
- When you type inside an existing sentence and the sentence boundaries haven't split or merged, `tryUpdateSegmentsInPlace` updates the `style` and `data-*` attributes on the existing DOM `<span>` nodes directly without replacing `innerHTML` or disturbing the cursor.

### Earlier-to-Later Bezier Arcs

When arcs are enabled on the rendered surface:
- Every arc points from the **earlier source sentence (`sourceIndex`)** to the **later target sentence (`targetIndex`)**, where `sourceIndex < targetIndex`.
- Arcs are drawn as smooth quadratic Bezier curves (`M sourceX sourceY Q controlX controlY targetX targetY`) in an SVG overlay positioned over the sentence spans.
- When you click a specific arc or sentence to inspect it, non-selected arcs dim to `0.3` opacity so the active path stands out clearly.
- Available arc modes in **Semantic Surface**:
  - **Pair select**: Click any two sentences to inspect the directional link from the earlier one to the later one.
  - **Top-k from sentence**: Click a sentence to show the top-k earlier sentences it attends back to.
  - **Global top-k**: Show the top-k strongest earlier-to-later sentence links across the entire document.
  - **Long distance relationship factor**: When toggled on, filters out short-range connections where `(targetIndex - sourceIndex) <= factor`, isolating long-range dependencies.

### Sentence View

Shows the `N x N` directional sentence matrix `S[p, q]` as an interactive heatmap:
- **Row `y`** = querying sentence
- **Column `x`** = attended-to sentence
- Clicking any cell `(x, y)` selects that sentence pair and opens **Token View** for those two sentences.

### Token View

Slices the max-pooled token attention matrix:

`TokenView[i, j] = max_{h, l} A[l, h, row_token_i, col_token_j]`

for the tokens belonging to the two selected sentences:
- **Rows** = tokens in `sentenceY` (the querying sentence)
- **Columns** = tokens in `sentenceX` (the attended-to sentence)
- Each cell shows the strongest attention weight from that row token to that column token across all layers and heads.

---

## What Is Cached in Memory and on Disk

Running a full causal pass and capturing per-layer tensors is `O(L * H * T^2)`. To avoid recomputing when you switch between tabs or modes on the same text:

- `FullContextAttentionAnalyzer._cached_state` (`AttentionDocumentState` in `backend/analysis.py`) caches the token list, pooled token attention tensor, sentence list, `sentence_boundaries_ids`, sentence matrix `S`, opportunity-corrected weights, and source character offsets for the current normalized text.
- `FullContextAttentionAnalyzer._cached_delta_capture` (`DeltaCaptureMetadata`) and `DeltaTensorStore` cache the memory-mapped `.npy` layer tensors and derived visualization JSON in `/tmp/attention-locator-delta-cache/<storage_key>/`.
- When the document text changes, the old cache is evicted and `gc.collect()` / `torch.cuda.empty_cache()` frees inference memory.

---

## Important Caveats to Keep in Mind

- **Prompt self-attention, not generation attribution**: The matrices show how tokens inside the document attend to earlier tokens in the same document pass, not which prompt tokens caused a generated completion token.
- **Text-defined sentence spans**: Sentence boundaries come from sentence tokenization aligned to subword tokens, not special sentence boundary tokens inside the model.
- **Max-pooling highlights peak head links**: Because `S[p, q]` takes the maximum over layers, heads, and token pairs, a single strong head-to-token connection between two sentences makes `S[p, q]` high.
- **Causal directionality**: Because the model is causal, later text can only attend to earlier text. Opportunity correction removes the automatic score advantage of early sentences, but the underlying links still flow from earlier context to later sentences.

---

## Persistent PostgreSQL + `pgvector` Residual Store

### Why Add a Vector Database?

`DeltaTensorStore` writes per-token `.npy` files to `/tmp` for the single document currently open in the UI. Those files are temporary and only cover one document at a time.

The PostgreSQL + `pgvector` integration (`backend/db.py`, `backend/vector_store.py`, `backend/schema.sql`) turns those captured residual stream states into a persistent, searchable vector store across documents and layers.

### Why Sentence-Mean-Pooled Vectors?

Storing a raw vector for every individual token across every layer (`L * T * hidden_size`) would generate thousands of high-dimensional rows per document. Instead, using the exact token spans `[token_start, token_end)` already aligned for each sentence, `_mean_pool` in `backend/vector_store.py` averages the token vectors belonging to sentence `s` at layer `l`:

```python
def _mean_pool(array: np.ndarray, start: int, end: int) -> np.ndarray:
    span = array[start:end]
    if span.shape[0] == 0:
        return np.zeros(array.shape[-1], dtype=np.float32)
    res = span.mean(axis=0).astype(np.float32)
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
```

For each sentence and each layer `0 .. L-1`, we store two vectors in `sentence_layer_states`:
- `'residual_input'`: the sentence's mean-pooled hidden state entering the attention block at layer `l`
- `'residual_after_attn'`: the sentence's mean-pooled hidden state after the attention block adds its delta back into the residual stream at layer `l`

Because every layer of the residual stream lives in the same `hidden_size`-dimensional vector space (zero-padded to `VECTOR(1536)` in `schema.sql`), cosine similarity across different layers and different documents is geometrically valid.

### Relational Schema (`backend/schema.sql`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    content_hash    TEXT UNIQUE NOT NULL,   -- SHA-256 of the document text
    model_name      TEXT NOT NULL,
    token_count     INT  NOT NULL,
    layer_count     INT  NOT NULL,
    head_count      INT  NOT NULL,
    hidden_size     INT  NOT NULL,
    head_dim        INT  NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

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

CREATE TABLE IF NOT EXISTS sentence_layer_states (
    id              SERIAL PRIMARY KEY,
    sentence_id     INT  NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    layer_index     INT  NOT NULL,
    vector_type     TEXT NOT NULL CHECK (vector_type IN ('residual_input', 'residual_after_attn')),
    embedding       VECTOR(1536),
    UNIQUE (sentence_id, layer_index, vector_type)
);

-- HNSW index for fast approximate nearest-neighbour search by cosine distance
CREATE INDEX IF NOT EXISTS idx_sentence_layer_states_hnsw
    ON sentence_layer_states
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### Example Queries & Endpoints

#### 1. Ingest a Document (`POST /vectors/ingest`)
Runs delta capture and sentence alignment on the document, computes the SHA-256 `content_hash` for idempotent upserts (`ON CONFLICT (content_hash) DO UPDATE`), and batch-inserts the `residual_input` and `residual_after_attn` vectors for every `(sentence, layer)` pair.

#### 2. Cross-Layer Nearest-Neighbor Search (`POST /vectors/search`)
Takes a query sentence (`sentenceIndex`) at `sourceLayer` from the current document, mean-pools its residual vector, and finds the closest sentence representations at `targetLayer` across all stored documents using `pgvector`'s cosine distance operator `<=>`:

```sql
SELECT
    s.id                         AS sentence_id,
    s.text                       AS sentence_text,
    s.sentence_index,
    s.attention_score,
    s.band,
    d.id                         AS document_id,
    d.content_hash,
    d.model_name,
    sls.layer_index,
    sls.embedding <=> $1         AS cosine_distance
FROM sentence_layer_states sls
JOIN sentences s ON sls.sentence_id = s.id
JOIN documents d ON s.document_id   = d.id
WHERE sls.vector_type = $2
  AND sls.layer_index = $3
ORDER BY cosine_distance ASC
LIMIT $4;
```

#### 3. Cross-Layer Representational Drift (`GET /vectors/drift/{sentence_id}`)
Measures how much a sentence's residual stream representation changes between every pair of adjacent layers `l` and `l + 1`. Lower cosine similarity between `l` and `l + 1` indicates higher representational drift at that layer:

```sql
SELECT
    a.layer_index                     AS from_layer,
    b.layer_index                     AS to_layer,
    1 - (a.embedding <=> b.embedding) AS cosine_similarity
FROM sentence_layer_states a
JOIN sentence_layer_states b
    ON  a.sentence_id = b.sentence_id
    AND b.layer_index = a.layer_index + 1
WHERE a.sentence_id = $1
  AND a.vector_type = 'residual_after_attn'
  AND b.vector_type = 'residual_after_attn'
ORDER BY cosine_similarity ASC;
```

#### 4. Document Layer Metadata (`GET /vectors/layers/{document_id}`)
Returns all stored sentence-layer state rows (without the raw 1536-float embedding arrays) for a given document to populate layer and sentence inspectors.
