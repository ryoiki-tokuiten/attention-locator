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
