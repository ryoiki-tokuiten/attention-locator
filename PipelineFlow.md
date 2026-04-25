**1. What text the model actually sees**
The model now sees the raw markdown source itself.

That means headings, list markers, code fences, inline code, math delimiters, tables, punctuation, and ordinary prose all go into the same input string. We only normalize line endings so the text is consistent. We do not strip markdown into “clean prose” first anymore.

So if you type:

```md
# Title

Here is `code`.

$$x^2$$
```

the model attends over the markdown form of that text, not a cleaned prose-only version.

**2. Who tokenizes it**
The model’s own tokenizer does.

So the sentence detector does not decide tokens. The tokenizer from the loaded language model takes the raw markdown string and breaks it into model-native subword tokens. Depending on the tokenizer, a “word” may become:
- one token
- several tokens
- tokens with leading-space markers
- punctuation-only tokens
- newline tokens
- code-ish fragments like backticks, fence markers, or math syntax as separate pieces

This tokenization is the true atomic unit the attention mechanism operates on.

**3. No chunking anymore**
The whole markdown document is sent in one pass, as long as it fits under the max token limit.

So conceptually the sequence is:

`raw markdown -> tokenizer -> full token sequence -> one forward pass`

There is no sentence batching, no sliding windows, no overlap merging anymore.

**4. What attention tensor we get from the model**
We ask the model to return attention from every layer during a normal forward pass over the input.

If:
- `L` = number of layers
- `H` = number of heads per layer
- `T` = token count

then the model gives attention tensors shaped like:

`L x H x T x T`

More explicitly, for each layer `l` and head `h`, you have a matrix:

`A[l, h, i, j]`

where:
- `i` is the query token position
- `j` is the key token position
- the value says how much token `i` is attending to token `j` inside that specific head and layer

Because this is a causal decoder model, attention is masked to the past. So token `i` can only attend to tokens `<= i`, not future tokens. That means the matrix is effectively lower-triangular.

So this is not “what answer token attended to the prompt while generating.”  
It is the self-attention structure of the prompt/document pass itself.

**5. How sentences are detected**
Tokens are model-defined. Sentences are not.

Sentences are defined by a sentence tokenizer used by the SAVIS logic. The system decodes the token stream back into text progressively, then uses a sentence tokenizer to propose sentence boundaries from that decoded text.

So there are really two different segmentations:
- token segmentation: from the model tokenizer
- sentence segmentation: from sentence-tokenization logic over decoded text

SAVIS then aligns those two worlds.

**6. How SAVIS aligns sentences to token spans**
This is the important part.

SAVIS walks through the decoded token stream and watches for newline-bearing tokens or the end of the sequence. At each such checkpoint, it:
1. decodes the text seen so far
2. takes the newly added chunk
3. runs sentence tokenization on that chunk
4. then advances token-by-token until the decoded slice contains each detected sentence
5. records the token boundary where that sentence ends

So each sentence becomes a token span:

`sentence_k = tokens[start_k : end_k]`

That gives SAVIS:
- a sentence list
- a list of token boundaries for those sentences

So sentence detection is text-based, but the result is converted into exact token index ranges.

**7. How token attention becomes sentence attention**
Now we have:
- raw token attention matrices from all heads and layers
- token spans for each sentence

From there the sentence matrix is built in three reductions.

For a sentence pair `(p, q)`:

- `p` is the querying sentence
- `q` is the attended-to sentence

First, for each head, SAVIS looks at the token rectangle covering:
- all query tokens in sentence `p`
- all key tokens in sentence `q`

So it inspects:

`A[l, h, i, j]` for all `i in sentence_p`, `j in sentence_q`

Then it reduces with max pooling.

The actual logic is:

1. Max over layers:
`A_head[h, i, j] = max_l A[l, h, i, j]`

2. For each head, max over all token pairs inside the sentence rectangle:
`S_head[h, p, q] = max_{i in p, j in q} A_head[h, i, j]`

3. Max over heads:
`S[p, q] = max_h S_head[h, p, q]`

So the final sentence-to-sentence matrix entry is:

`S[p, q] = max_{h, l, i in p, j in q} A[l, h, i, j]`

That is a very specific choice.

It does **not** mean:
- average attention between the two sentences
- total mass between the two sentences
- attention from all heads equally combined

It means:
“what is the single strongest token-to-token link between these two sentences, across any layer and any head?”

**8. What the sentence matrix means**
Rows are query sentences. Columns are attended-to sentences.

So `S[p, q]` means:

“how strongly sentence `p` attends to sentence `q` under the max-pooled SAVIS summary”

Because the model is causal:
- later sentences can look back at earlier sentences
- earlier sentences cannot look forward to later ones

So the matrix is directional.

**9. How one sentence gets its final scalar score**
The colored sentence weight in the main Rendered tab is not a raw matrix value. It is derived from the sentence matrix, but it is now opportunity-corrected to reduce the built-in causal bias toward early sentences.

The first step is still the same: for sentence `j`, compute the amount of sentence-level attention it receives from all other sentences:

`I_j = sum_i S(i, j) - S(j, j)`

So:
- sum the whole column for sentence `j`
- remove self-attention on the diagonal

If we stopped here, early sentences would still tend to win simply because more later tokens are allowed to attend to them.

So the system then estimates how many later tokens even had the *opportunity* to cite sentence `j`.

Let:
- `b_k` = the token boundary index where sentence `k` ends
- `n` = the number of usable sentences
- `T = b_n` = the total usable token count in the analyzed document

Then the later-token opportunity count for sentence `j` is:

`O_j = max(T - b_{j+1}, 1)`

This means:
- look at where sentence `j` ends
- count how many usable tokens still exist after it
- clamp the count to at least `1` so the denominator never becomes zero

So `O_j` is literally “how many later tokens were even allowed to look back at this sentence?”

The system then computes the document-average opportunity count:

`O_bar = max((1 / n) * sum_j O_j, 1)`

This average is important. If we only divided by `O_j`, the corrected scores would become artificially tiny for many documents. Instead, the system rescales everything back by the average opportunity count so the magnitudes stay in a comparable range.

The final raw sentence score is therefore:

`R_j = I_j * (O_bar / O_j)`

Equivalently:

`R_j = (sum_i S(i, j) - S(j, j)) * (O_bar / O_j)`

This does three useful things at once:
- sentences with *more than average* opportunity get scaled down
- sentences with *less than average* opportunity get scaled up
- the average overall scale is preserved instead of collapsing toward zero

So the system is no longer using plain incoming attention. It is using an opportunity-corrected citation score.

If sentence-boundary token information is missing or truncated, the system falls back to the simpler uncorrected form:

`R_j = I_j`

**10. How those scores are normalized**
Those corrected raw scores are still document-relative.

If all corrected scores are identical, every sentence is assigned the neutral midpoint:

`W_j = 0.5`

Otherwise the system applies min-max normalization:

`W_j = (R_j - min_k R_k) / (max_k R_k - min_k R_k)`

So the full scoring pipeline is:

1. Build the sentence matrix `S`
2. Compute incoming sentence citation `I_j`
3. Compute later-token opportunity count `O_j`
4. Compute average opportunity `O_bar`
5. Produce corrected raw score `R_j = I_j * (O_bar / O_j)`
6. Min-max normalize `R_j` into `W_j`

So a displayed score near `1.0` now means:
- this sentence received unusually strong incoming attention
- after correcting for how many later tokens were even able to attend to it

It still does **not** mean:
- the model gave this sentence “100% importance”
- the sentence is globally important outside the current document
- the model necessarily used it in a task-specific causal sense

**11. How low / medium / high bands are assigned**
After normalization, the scores are bucketed into bands using document-relative percentiles.

So `high`, `medium`, and `low` are not universal categories. They are shaped by the score distribution of the current document. If the document is very flat, the system forces a more readable middle distribution instead of making everything look dead.

**12. What the token view is actually showing**
When you pick a sentence pair, the system slices the token-level matrix for just those two sentence spans.

But the token matrix you see is already pooled:
- max over layers
- max over heads

So token view is **not** raw per-layer, per-head attention.

It shows:

`TokenView[i, j] = max_{h, l} A[l, h, row_token_i, col_token_j]`

And because rows are queries and columns are attended-to tokens, a cell means:

“how much this row token attends to that column token under the max-pooled summary”

**13. What the sentence view is actually showing**
Sentence view shows the sentence matrix `S[p, q]`.

That means:
- row = querying sentence
- column = attended-to sentence
- value = strongest cross-token attention link between those two sentences, after max over layers and heads

So the sentence heatmap is a compressed map of directional inter-sentence attention.

**14. What the Rendered tab is actually showing**
Rendered tab is not drawing raw attention matrices.

It does this instead:
1. take the sentence scores and sentence ranges from the backend
2. render the markdown normally into HTML
3. keep source offsets that tie rendered text back to the original markdown source
4. find visible text ranges whose source overlaps a scored sentence
5. wrap those overlaps in styled spans
6. color them using the normalized sentence score

So the Rendered tab is a projection of sentence-level salience back onto the rendered markdown surface.

That means:
- it is sentence-level, not token-level
- it is based on sentence source ranges, not browser sentence guessing
- it can highlight text even when the sentence is split across inline markup
- it is not a literal display of every model token the model saw

For example, markdown syntax characters like `#`, backticks, `**`, or math delimiters may influence the model’s attention because the model sees them, but they may not appear the same way in the rendered surface. So the Rendered tab is a visual mapping of sentence salience onto the human-facing markdown render, not a raw token trace.

**15. How source ranges are recovered**
After SAVIS has its sentence strings, the system maps each sentence back onto the original markdown source by normalizing whitespace and matching sentence text against the source text.

That gives each sentence:
- a source start offset
- a source end offset

Those offsets are what let the rendered preview highlight backend-defined sentence regions without re-segmenting text in the browser.

**16. What is cached**
Once the document is analyzed, the system keeps the reduced state for that exact text:
- token list
- pooled token attention matrix
- sentence list
- sentence boundaries
- sentence matrix
- final sentence weights
- source offsets

So changing tabs does not require recomputing everything. If the text changes, the old state is dropped and memory is cleared.

**17. What this means conceptually**
Right now the pipeline is:

`raw markdown`
`-> model tokenizer`
`-> full causal self-attention over all tokens`
`-> SAVIS sentence boundary alignment`
`-> max over layers`
`-> max over token pairs inside sentence rectangles`
`-> max over heads`
`-> incoming sentence citation`
`-> later-token opportunity correction`
`-> normalized sentence weights`
`-> projection back onto rendered markdown by source offsets`

**18. The most important caveats**
This is the part people usually blur, so it is worth being precise.

1. This is prompt self-attention, not answer-conditioned attribution.
2. Sentence boundaries are not model-native; they are sentence-tokenizer-defined and then aligned to tokens.
3. Sentence scores are driven by max pooling, so one strong head/layer/token link can dominate.
4. The Rendered tab is a sentence-salience overlay, not a literal token-by-token depiction of the exact visible page.
5. Because the model is causal, the visualization still mostly tells you which earlier text later text looks back to, even though the opportunity correction reduces the most obvious front-loaded bias.


--------------------------


DBMS integration:

## Background

The existing `DeltaTensorStore` captures per-layer, per-head tensors from every forward pass and writes them to memory-mapped `.npy` files in `/tmp`. The data is rich but ephemeral — it is discarded when the server restarts or when the user switches documents. The DBMS integration replaces this volatile temp-file approach with a persistent PostgreSQL + pgvector backend that makes all captured vectors searchable.

---

## What vectors we already have (from `delta_capture.py`)

For every document analysis, per layer `l` (18 layers in Gemma-270M), the system captures:

| Tensor Name | Shape | What it is |
|---|---|---|
| `residual_inputs` | `[L, T, H_dim]` | The hidden state entering each transformer block |
| `attention_weights` | `[L, N_heads, T, T]` | The raw per-head attention matrix |
| `value_states` | `[L, N_heads, T, head_dim]` | The projected value vectors before mixing |
| `head_contexts` | `[L, N_heads, T, head_dim]` | The per-token output of each individual head (Av) |
| `head_projected_deltas` | `[L, N_heads, T, H_dim]` | Each head's contribution projected into full residual space |
| `attention_deltas_post_norm` | `[L, T, H_dim]` | The full layer attention output after layer norm |
| `residual_after_attention` | `[L, T, H_dim]` | The residual stream state after attention block |

> **Key insight:** Each row `[l, t, :]` in `residual_inputs` is a `H_dim`-dimensional vector representing a single token's full hidden state entering a specific transformer layer. These are exactly what we want to store and query.

---

## The Core DBMS Concept

The proposal is a **Neural Vector Registry** — a PostgreSQL database that:

1. **Stores** sentence-averaged residual and delta vectors from each layer and head.
2. **Indexes** them using `pgvector`'s HNSW (Hierarchical Navigable Small World) approximate nearest-neighbour index.
3. **Answers** queries of the form: *"Find all sentence embeddings in the database whose layer-2, head-3 head context vector is closest to this query vector."*

This is fundamentally a **DBMS primitive** — a nearest-neighbour join on a high-dimensional relation — applied to a ML pipeline.

---

## Relational Schema

```sql
-- A document that was analyzed
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    content_hash    TEXT UNIQUE NOT NULL,   -- SHA-256 of markdown text
    model_name  TEXT NOT NULL,
    token_count INT  NOT NULL,
    layer_count INT  NOT NULL,
    head_count  INT  NOT NULL,
    hidden_size INT  NOT NULL,
    head_dim    INT  NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- A single sentence extracted from a document
CREATE TABLE sentences (
    id              SERIAL PRIMARY KEY,
    document_id     INT  REFERENCES documents(id) ON DELETE CASCADE,
    sentence_index  INT  NOT NULL,
    text            TEXT NOT NULL,
    token_start     INT  NOT NULL,    -- inclusive
    token_end       INT  NOT NULL,    -- exclusive
    attention_score FLOAT,            -- opportunity-corrected W_j score
    band            TEXT,             -- 'high' / 'medium' / 'low'
    UNIQUE (document_id, sentence_index)
);

-- The core vector table: one row per (sentence, layer, vector_type)
-- Stores sentence-averaged vectors for residual states and deltas
CREATE TABLE sentence_vectors (
    id              SERIAL PRIMARY KEY,
    sentence_id     INT  REFERENCES sentences(id) ON DELETE CASCADE,
    layer_index     INT  NOT NULL,
    vector_type     TEXT NOT NULL,   -- 'residual_input' | 'residual_after_attn' | 'attn_delta_post_norm'
    embedding       VECTOR(1152)     -- hidden_size for Gemma-270M
);

-- Per-head vector table: one row per (sentence, layer, head, vector_type)
-- Stores head-specific context and projected delta vectors
CREATE TABLE sentence_head_vectors (
    id              SERIAL PRIMARY KEY,
    sentence_id     INT  REFERENCES sentences(id) ON DELETE CASCADE,
    layer_index     INT  NOT NULL,
    head_index      INT  NOT NULL,
    vector_type     TEXT NOT NULL,   -- 'head_context' | 'value_state' | 'projected_delta'
    embedding       VECTOR(64)       -- head_dim for Gemma-270M
);

-- HNSW Indexes for ANN search
CREATE INDEX ON sentence_vectors        USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON sentence_head_vectors   USING hnsw (embedding vector_cosine_ops);
```

### Why Sentence-Averaged?

Storing every token individually would produce `L * T * H_dim` float16 values per document — gigabytes per document. Instead, tokens belonging to each sentence (using `token_start:token_end` bounds already computed by SAVIS) are **mean-pooled** into a single representative vector before insertion. This brings the storage down to `N_sentences * L * 3` rows in `sentence_vectors`, which is entirely manageable.

---

## Query Examples (The Academic Payoff)

### Q1: Find the sentences whose layer-2 residual state is closest to a given sentence
```sql
SELECT
    s.text,
    d.model_name,
    sv.embedding <=> %(query_vec)s AS cosine_distance
FROM sentence_vectors sv
JOIN sentences s ON sv.sentence_id = s.id
JOIN documents d ON s.document_id = d.id
WHERE sv.layer_index = 2
  AND sv.vector_type = 'residual_input'
ORDER BY cosine_distance ASC
LIMIT 10;
```

### Q2: Find all sentences where head 3 of layer 2 produced a head_context vector closest to a query vector (the exact use-case you described)
```sql
SELECT
    s.text,
    s.attention_score,
    shv.embedding <=> %(query_vec)s AS distance
FROM sentence_head_vectors shv
JOIN sentences s ON shv.sentence_id = s.id
WHERE shv.layer_index = 2
  AND shv.head_index  = 3
  AND shv.vector_type = 'head_context'
ORDER BY distance ASC
LIMIT 10;
```

### Q3: Cross-layer drift — find sentences where the residual state changed the most between layer 0 and the final layer (uses a window function join)
```sql
WITH early AS (
    SELECT sentence_id, embedding AS e0 FROM sentence_vectors
    WHERE layer_index = 0 AND vector_type = 'residual_input'
),
late AS (
    SELECT sentence_id, embedding AS eN FROM sentence_vectors
    WHERE layer_index = (SELECT MAX(layer_index) FROM sentence_vectors)
      AND vector_type = 'residual_after_attn'
)
SELECT s.text, 1 - (e.e0 <=> l.eN) AS cosine_similarity
FROM early e
JOIN late l  ON e.sentence_id = l.sentence_id
JOIN sentences s ON s.id = e.sentence_id
ORDER BY cosine_similarity ASC
LIMIT 10;
```

This query finds sentences where the residual stream **drifted most** from the input to the output — a direct measure of how much attention transformed that sentence's representation.

---

## New API Endpoints to Add

```
POST /vectors/ingest          — Run analysis + ingest vectors for a document
POST /vectors/search          — ANN search given a query sentence and params
GET  /vectors/layers/{doc_id} — Retrieve all layer vectors for a document (for UI)
```

The `/vectors/search` endpoint accepts:
- `query_text`: a sentence string to embed and search with
- `layer_index`: which layer to search in
- `head_index` (optional): if provided, searches `sentence_head_vectors`
- `vector_type`: which tensor type to compare against
- `limit`: top-k results

## Academic Talking Points for DBMS Course

| Concept | Where it appears |
|---|---|
| **3NF Relational Design** | `documents` → `sentences` → `sentence_vectors` decomposition |
| **Foreign Key Constraints + Cascades** | `ON DELETE CASCADE` throughout the hierarchy |
| **Composite Unique Constraints** | `UNIQUE(document_id, sentence_index)` |
| **High-Dimensional Indexing** | HNSW index instead of B-tree, and why B-tree fails for vectors |
| **Approximate Nearest Neighbour** | Why exact KNN is O(N*D) and HNSW gives O(log N) |
| **Aggregation & Window Functions** | Cross-layer drift query using CTE + self-join |
| **Vector Operators** | `<=>` cosine distance, `<->` L2, `<#>` inner product — and when to use which |
| **Idempotent Ingestion** | `ON CONFLICT DO NOTHING` via `content_hash` |
| **Connection Pooling** | `asyncpg` + FastAPI lifespan for the DBMS course |

---

# Concepts of DAA (Data Structures and Algorithms) Used

The integration of the SAVIS (Self-Attention Visualizer) pipeline with the persistent neural registry is heavily grounded in classical and modern Data Structures and Algorithms (DAA). The codebase orchestrates a complex pipeline that transforms unstructured markdown text into high-dimensional vector representations, leveraging optimal data structures to guarantee performance, modularity, and scalability. 


## 1. High-Dimensional Graph Search (Hierarchical Navigable Small World - HNSW)

**Concept:** Approximate Nearest Neighbor (ANN) Search, Graph Data Structures
**Location:** `backend/schema.sql`

When scaling vector databases, exact K-Nearest Neighbors (KNN) requires an exhaustive scan of the database, yielding an unacceptable time complexity of $O(N \times D)$ where $N$ is the number of vectors and $D$ is the dimensionality (1536 in our schema). 

To solve this, the infrastructure employs **HNSW (Hierarchical Navigable Small World)** graphs, implemented natively via `pgvector`. 

```sql
-- HNSW index for approximate nearest-neighbour search using cosine distance.
-- m=16 and ef_construction=64 are the pgvector defaults; good for most workloads.
CREATE INDEX IF NOT EXISTS idx_sentence_layer_states_hnsw
    ON sentence_layer_states
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

**Algorithmic Deep Dive:**
HNSW is a multi-layered proximity graph. 
- **Data Structure:** The bottom layer contains all vectors. Higher layers contain exponentially fewer vectors (like a skip list). Nodes are connected to their nearest neighbors.
- **Search Algorithm:** The search starts at the top (sparsest) layer, greedily traversing towards the query vector. Once a local minimum is reached, the search drops to the next denser layer, repeating until the base layer is reached.
- **Complexity:** Reduces the search complexity from $O(N \times D)$ to **$O(\log N)$**.
- **Distance Metric:** We use `vector_cosine_ops` (`<=>` operator in PostgreSQL), which calculates the cosine distance $1 - \frac{A \cdot B}{||A|| ||B||}$.

## 2. Dynamic Programming / Memoization for State Management

**Concept:** Memoization, Caching, Overlapping Subproblems
**Location:** `backend/analysis.py` (`_cached_state`, `_cached_delta_capture`)

Extracting full causal self-attention matrices from a Transformer is extraordinarily expensive. The time complexity of self-attention is $O(L \times H \times T^2)$, where $L$ is layers, $H$ is heads, and $T$ is the number of tokens. For large documents, this computation easily dominates the pipeline.

**Implementation:**
The `FullContextSAVISAnalyzer` class implements strict memoization to avoid recomputing the $O(T^2)$ matrices if the user requests different views of the exact same text.

```python
def _compute_state(self, markdown_text: str) -> SAVISDocumentState:
    prepared_text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Memoization hit: Return cached $O(T^2)$ tensor computations
    if self._cached_state and self._cached_state.cache_key == prepared_text:
        return self._cached_state
        
    # Memoization miss: Evict old cache, free GPU VRAM, compute new
    if self._cached_state is not None:
        self._cached_state = None
        self._release_inference_memory()
        
    # ... heavy torch model inference ...
```

**Algorithmic Deep Dive:**
By storing the complex `SAVISDocumentState` (which acts as a memoization table), the frontend can rapid-fire requests (e.g., swapping between token view and sentence view) in $O(1)$ amortized time instead of triggering $O(T^2)$ GPU recalculations.

## 3. Dimensionality Reduction via Mean Pooling (Aggregation Algorithms)

**Concept:** Vector Aggregation, Matrix Slicing
**Location:** `backend/vector_store.py` (`_mean_pool`)

The raw tensor output for a document is shaped `[L, T, D]`. Storing every single token (`T`) would explode the database size. The algorithm reduces this by pooling token vectors into sentence boundaries $O(K \times D)$, where $K$ is the number of tokens in a specific sentence.

```python
def _mean_pool(array: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Mean-pool rows [start:end] of `array` along axis 0.
    Returns a 1-D float32 vector of shape (array.shape[-1],).
    """
    span = array[start:end]
    if span.shape[0] == 0:
        return np.zeros(array.shape[-1], dtype=np.float32)
    res = span.mean(axis=0).astype(np.float32)
    # Sanitize inf/nan which pgvector rejects
    return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
```

**Algorithmic Deep Dive:**
- Slicing `array[start:end]` is a constant-time $O(1)$ view operation in numpy.
- The `.mean(axis=0)` iterates over the $K$ rows, executing in $O(K \times D)$ time.
- This creates a massive memory optimization, converting an $O(T)$ storage requirement into an $O(S)$ requirement (where $S$ is the number of sentences, $S \ll T$).

## 4. O(N log N) Sorting and Adaptive Selection

**Concept:** Sorting Algorithms, Percentile Calculation
**Location:** `backend/analysis.py` (`percentile`, `assign_score_bands`)

To visually color sentences on the frontend, the system doesn't use static thresholds (which fail if the model's attention distribution is flat). Instead, it dynamically calculates percentile cutoffs using a sorting algorithm.

```python
def percentile(values: Sequence[float], ratio: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values) # O(N log N)
    position = (len(ordered) - 1) * ratio
    lower_index = floor(position)
    upper_index = ceil(position)

    if lower_index == upper_index:
        return ordered[lower_index]

    # Linear interpolation
    progress = position - lower_index
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    return lower_value + ((upper_value - lower_value) * progress)
```

**Algorithmic Deep Dive:**
The `sorted(values)` call utilizes Python's Timsort algorithm. Timsort is a hybrid sorting algorithm derived from merge sort and insertion sort, giving a worst-case and average-case time complexity of **$O(N \log N)$**. This allows the system to establish statistical boundaries (e.g., 78th percentile for 'high' band, 38th for 'medium') robustly, no matter the length of the document.

## 5. Tensor Algebra and Matrix Multiplication (Einsum)

**Concept:** Linear Algebra, Optimized Hardware Operations
**Location:** `backend/analysis.py` (Inside `build_attention_hook`)

When intercepting transformer layers to calculate `head_projected_delta`, the system avoids naive iterative nested loops and utilizes highly optimized Einstein summation (`torch.einsum`).

```python
head_projected_delta = self.torch.einsum(
    "bthd,hod->btho",
    head_context,
    projection_blocks,
).contiguous()
```

**Algorithmic Deep Dive:**
Naive matrix multiplication has a complexity of $O(N^3)$. `torch.einsum` evaluates the algebraic notation and delegates to highly optimized BLAS (Basic Linear Algebra Subprograms) libraries on the CPU or cuBLAS on the GPU. It optimizes memory access patterns, maximizes cache locality, and leverages SIMD/Tensor cores, reducing the hidden constant factors drastically while processing the 4D tensors `[batch, token, head, dim]`.

## 6. O(1) Hash-based Idempotency

**Concept:** Cryptographic Hashing, Set/Existence Checking
**Location:** `backend/vector_store.py` (`compute_content_hash`)

Database insertions must be idempotent so that re-analyzing the same document doesn't duplicate vectors.

```python
def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
```

This pairs with the SQL UPSERT algorithm:
```sql
ON CONFLICT (content_hash) DO UPDATE
```

**Algorithmic Deep Dive:**
- SHA-256 processes the string in $O(N)$ time where $N$ is the string length.
- The resulting hash is indexed in the PostgreSQL B-Tree (`content_hash TEXT UNIQUE NOT NULL`).
- Checking if a document already exists becomes an **$O(\log M)$** B-Tree lookup (or effectively $O(1)$ if hash indexed), preventing $O(M)$ sequential scans across $M$ stored documents.

## 7. The Algorithmic Pipeline Flow

The entire architecture is modularized into a strict, unidirectional data structure pipeline:

1. **Input Phase:** Raw Text $\rightarrow$ $O(N)$ Normalization $\rightarrow$ SHA-256 Hashing.
2. **Tokenizer Phase:** Text mapped to Integer Array of tokens (BPE/SentencePiece algorithm).
3. **Inference Phase (The Bottleneck):** $O(T^2 \cdot L \cdot H)$ Forward pass generating intermediate state tensors.
4. **Hook Phase:** Extracting $O(T \cdot D)$ residual states per layer via PyTorch hooks.
5. **Alignment Phase:** $O(T)$ linear scan mapping sentence strings back to source offsets (`sentence_source_offsets`).
6. **Scoring Phase:** $O(S^2)$ reduction of Token Attention Matrices into Sentence Attention Matrices, followed by Opportunity Correction.
7. **Banding Phase:** $O(S \log S)$ Timsort to assign visual heat bands.
8. **Pooling Phase:** $O(S \cdot K \cdot D)$ Mean-pooling token deltas into sentence representations.
9. **Persistence Phase:** Database bulk-insert and $O(\log M)$ graph insertion into the HNSW index.

This modular pipeline perfectly isolates computational boundaries. The math engine (`analysis.py`), the state storage (`delta_capture.py`), and the persistence layer (`vector_store.py`) act as independent algorithmic nodes, passing heavily optimized, serialized data structures between each other.