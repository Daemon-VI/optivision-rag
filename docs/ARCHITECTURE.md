# Architecture

## Why the pipeline is shaped this way

The project constraint from the abstract is precise: *reduce the memory used by
VLM document retrieval without changing the model*. That rules out a lot of
otherwise-reasonable designs — fine-tuning a smaller projection head, distilling
ColPali, training a learned pooler. All of them would make the results
un-transferable to someone already running ColPali in production.

So the whole contribution sits between "the model emitted vectors" and "the
vectors went into the index". Everything upstream and downstream is stock.

```
                          OUR SCOPE
                    ┌───────────────────┐
 page ─► encoder ─► │ prune ─► quantize │ ─► index ─► MaxSim ─► results
         (stock)    └───────────────────┘   (stock)   (stock)
                                              ▲
 query ─► encoder ────────────────────────────┘
          (stock, full precision)
```

## Data flow and types

Three dataclasses mark the three states a page passes through
(`src/optivision/types.py`):

| type | holds | produced by |
|---|---|---|
| `PageEncoding` | `[n_tokens, dim]` float32, a `PatchGrid`, which tokens are text | `encoders/` |
| `PrunedPage` | surviving vectors, keep-mask, saliency map, per-stage stats | `pruning/` |
| `CompressedPage` | bit-packed `uint8` codes + byte accounting | `compression/` |

`CompressedPage` carries `n_tokens_before` all the way to the index so that
compression ratios are computed against what the *unmodified* model would have
stored, not against the post-pruning count. That distinction is the difference
between an honest 100× and a flattering 32×.

## The patch grid, and why it is fiddly

Spatial pruning has to answer "which pixels does vector *i* come from?". That
mapping is not the same across models:

**Single-image layouts** (ColPali/PaliGemma, ColSmol with splitting off) emit one
contiguous run of `rows × cols` image tokens in row-major order. Easy.

**Tiled layouts** (Idefics3/ColSmol default) cut the page into a `cr × cc` grid of
tiles, encode each to `s × s` patches, emit them as equal-length runs in tile order,
then append one more run for a thumbnail of the whole page. ColSmol-256M on an A4
page produces **13 runs of 64 tokens** — twelve 4×3 tiles plus the thumbnail.

Treating those 832 tokens as one flat rectangle (the obvious `isqrt` guess gives
32×26) maps saliency onto the wrong patches entirely, and it fails *silently*: the
index still builds, the search still returns pages, the quality is just quietly worse.
`ColVLMEncoder._grid_for` therefore detects the run structure, stitches the tiles into
a true `32×24` page grid, and sets the thumbnail run aside to be kept verbatim.

`optivision explain` exists largely to make this checkable by eye — if the green
cells do not sit on the ink, the grid mapping is wrong.

## Pruning

### Stage 1 — spatial (`pruning/saliency.py`, `pruning/spatial.py`)

Per grid cell, on a `cols*8 × rows*8` grayscale copy of the page:

```
paper    = 95th percentile of the page's intensity   (not "white")
ink      = mean over the cell of max(paper - pixel, 0)
edge     = mean over the cell of |∂x| + |∂y|
saliency = (0.6 · unit(ink) + 0.4 · unit(edge)) / 1.0
```

`unit()` scales by the 99th percentile rather than the max, so one dust speck or
punch-hole cannot squash a whole page toward zero.

Both signals are needed. Ink alone drops faint handwritten annotations; edges alone
keep scanner noise on empty margins.

Two selection modes:

- **threshold** (`blank_threshold`) — keep whatever carries ink. Token count adapts to
  the page, which is the honest behaviour: a dense table should cost more than a title
  page. Followed by a dilation step, because glyphs bleed across patch boundaries.
- **budget** (`keep_ratio`) — keep exactly the top *k* fraction. Needed for ablation
  curves and index-size guarantees. **Dilation is deliberately disabled here**: a
  scattered top-25% mask nearly triples under one dilation step, which would silently
  blow past the budget the mode exists to enforce.

`min_keep` floors both — a nearly blank page must stay retrievable, and an empty
vector list would break MaxSim.

### Stage 2 — redundancy (`pruning/redundancy.py`)

Greedy single-pass clustering in descending saliency order, so the most informative
patch becomes its cluster's representative. Vectors within `cosine ≥ 0.92` of a
representative join it; each cluster collapses to its renormalised mean.

The justification is specific to late interaction. MaxSim scores a query vector by
`max_j q·d_j`. If two document vectors are within cosine 0.92, the max over the pair
and the similarity to their normalised mean differ by an amount bounded by their
angular spread — small. So collapsing a tight cluster barely moves the score while
removing real bytes. This is *not* true for a single-vector retriever, where averaging
destroys information; it is a property of the max.

## Compression (`compression/binary.py`)

`sign(x)` per dimension, bit-packed: 128 dims → 16 bytes, exactly 32×.

Asymmetric scoring by default — query in float32, documents as ±1:

```
score(q, d) = Σ_i max_j  q_i · sign(d_j)
```

The symmetric alternative (binarise the query too, score by popcount) is faster but
throws away the one side we can afford to keep precise. A query has ~20 vectors; a
corpus has millions. `maxsim_hamming` implements it anyway so the ablation can price
the difference.

`int8` scalar quantization is included as the 4× middle ground — not because it is a
good answer here, but because a table with only "1×" and "32×" in it does not show
where quality actually starts to bend.

## Indexing

Two backends, deliberately:

**`NumpyIndex`** — exact brute-force MaxSim over the packed codes. All quality numbers
in the benchmark come from here, so that a change in the score can only be caused by
pruning or quantization, never by an ANN graph missing a neighbour. Pages are stored as
one concatenated code matrix plus an offsets array, and scored with a single matmul
followed by `np.maximum.reduceat` over the page segments — which is what makes an
exhaustive index fast enough to be the reference.

**`QdrantIndex`** — one point per page, whose vector is the whole list of surviving
patch vectors, compared with a native `MAX_SIM` multivector comparator and
`BinaryQuantization` inside the engine. This is the deployment-shaped path. Storage
figures reported by `stats()` still come from the pipeline's own byte accounting, so
they stay comparable between backends.

## The benchmark harness (`bench.py`)

Encoding dominates runtime (~30 s/page on CPU), and every ablation row needs the same
vectors. So `EncodedCorpus` encodes the corpus **once**, caches the encodings plus a
512 px grayscale copy of each page (spatial pruning needs pixels), and every variant
replays over that cache. Nothing but the compression settings differ between rows.

Quality is reported twice, because the two answer different questions:

- **absolute** (nDCG@5, Recall@1) — is the system any good?
- **Kendall tau vs. the baseline's own ranking** — did compression change the model's
  mind? This stays informative even when the absolute metric saturates.

## Known sharp edges

- `SyntheticEncoder` falls back to pixel hashing for pages missing from its word
  layout. That fallback is legitimate for arbitrary images but catastrophic-and-silent
  if page ids drift, so it raises on a missing layout file and warns loudly on a
  missing page. A test pins the id convention.
- Sign-quantising a zero vector invents an all-`-1` direction that is pure noise.
  Encoders must not emit zero vectors; the synthetic one adds a tiny fixed "blank
  paper" component so blank cells have a defined direction.
- `np.maximum.reduceat` returns garbage for empty segments. `NumpyIndex.score_all`
  masks those pages out rather than trusting the result.
