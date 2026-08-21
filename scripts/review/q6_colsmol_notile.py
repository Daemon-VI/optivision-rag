"""The resolution confound. ColSmol tiles an A4 page into 12 x 512px crops and
can read an 11pt code; ColPali sees the page at 448x448 and plausibly cannot.
E3's 'ColPali loses nothing to one-bit codes' may therefore be a statement
about a retriever that is not reading the discriminative evidence. Test: make
ColSmol equally blind (image splitting off -> one 512px view, 64 tokens) on the
same pages and see whether *its* one-bit loss collapses too. Same encoder, same
corpus, only the resolution moves.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Users/Rishi/optivision-rag/src")
from optivision.bench import EncodedCorpus, _encode_queries_cached  # noqa: E402
from optivision.config import Config  # noqa: E402
from optivision.corpus import load_queries  # noqa: E402
from optivision.encoders import get_encoder  # noqa: E402
from optivision.metrics import ndcg_at_k, rank_correlation  # noqa: E402

ROOT = Path("C:/Users/Rishi/optivision-rag")
SCR = Path(__file__).resolve().parent
os.chdir(ROOT)
cache = SCR / "colsmol_notile.npz"
cfg = Config()

if not cache.exists():
    enc = get_encoder(cfg.encoder)
    ip = enc.processor.image_processor
    print("image processor:", type(ip).__name__, "do_image_splitting =", getattr(ip, "do_image_splitting", "?"),
          "size =", getattr(ip, "size", "?"), "max_image_size =", getattr(ip, "max_image_size", "?"), flush=True)
    ip.do_image_splitting = False
    t0 = time.time()
    corpus = EncodedCorpus.build("data/corpus/pdfs", enc, cfg, progress=lambda i, r: print(f"  {i}/60 {r.page_id} {time.time()-t0:.0f}s", flush=True) if i % 10 == 0 else None)
    print("tokens on page 0:", corpus.encodings[0].n_tokens, "grid", corpus.encodings[0].grid.rows, "x", corpus.encodings[0].grid.cols,
          "text tokens", corpus.encodings[0].text_token_index.size, flush=True)
    corpus.save(cache)
    qids, texts, qrels = load_queries("data/corpus/queries.json")
    _encode_queries_cached(texts, cfg, enc, cache.with_suffix(".queries.npz"))
corpus = EncodedCorpus.load(cache)
qids, texts, qrels = load_queries("data/corpus/queries.json")
zq = np.load(cache.with_suffix(".queries.npz"), allow_pickle=True)
queries = [zq[f"q_{i}"] for i in range(len(texts))]
rel = json.loads((ROOT / "data/corpus/queries.json").read_text())
family = np.array([r["type"] for r in rel])
ids = [e.ref.page_id for e in corpus.encodings]
pages = [e.embeddings for e in corpus.encodings]
print(f"\nColSmol, no tiling: {len(pages)} pages x {pages[0].shape[0]} tokens")


def per_query(mats, q_fn=lambda q: q):
    nd, r1, od = [], [], []
    for qi, q in enumerate(queries):
        qq = q_fn(q)
        sc = np.array([float((qq @ d.T).max(1).sum()) for d in mats])
        ranked = [ids[i] for i in np.argsort(-sc)]
        nd.append(ndcg_at_k(ranked, qrels[qids[qi]], 5)); r1.append(ranked[0] in qrels[qids[qi]]); od.append(ranked)
    return np.array(nd), np.array(r1, float), od


allv = np.concatenate(pages); D = allv.shape[1]; mu = allv.mean(0)
p_pos = (allv > 0).mean(0)
h = -(p_pos * np.log2(np.clip(p_pos, 1e-9, 1)) + (1 - p_pos) * np.log2(np.clip(1 - p_pos, 1e-9, 1)))
ev = np.linalg.eigvalsh(np.cov((allv - mu).T))
print(f"geometry: ||mu|| {np.linalg.norm(mu):.3f}  dead bits (H<0.5) {(h<0.5).sum()}  participation ratio {ev.sum()**2/(ev**2).sum():.1f}")

rng = np.random.default_rng(7)
_, V = np.linalg.eigh(np.cov((allv - mu).T)); V = V[:, ::-1]
def itq(v, iters=50):
    R, _ = np.linalg.qr(rng.standard_normal((v.shape[1], v.shape[1])))
    for _ in range(iters):
        B = np.sign(v @ R); U, _, Wt = np.linalg.svd(B.T @ v); R = (U @ Wt).T
    return R
Vq = V @ itq((allv - mu) @ V)

base = per_query(pages)
rows = [("float32", base),
        ("sign(d)", per_query([np.sign(p) for p in pages])),
        ("sign(ITQ(d-mu))", per_query([np.sign((p - mu) @ Vq) for p in pages], q_fn=lambda q: q @ Vq)),
        ("int8", per_query([(np.clip(np.round(p / 0.5 * 127), -127, 127) * 0.5 / 127).astype(np.float32) for p in pages]))]
print(f"\n{'codec':<18} nDCG@5  retain   R@1   tau60 | precise nDCG/R@1 | topical nDCG/R@1")
for name, (nd, r1, od) in rows:
    tau = np.mean([rank_correlation(a, b, pool=ids) for a, b in zip(base[2], od)])
    pm, tm = family == "precise", family == "topical"
    print(f"{name:<18} {nd.mean():.4f} {100*nd.mean()/base[0].mean():6.1f}%  {r1.mean():.3f}  {tau:.3f} |  {nd[pm].mean():.3f} / {r1[pm].mean():.3f}   |  {nd[tm].mean():.3f} / {r1[tm].mean():.3f}")
print("\nreference, tiled ColSmol (E1): float 0.7823/R@1 0.569 (precise 0.745/0.483); sign 87.9%, R@1 0.417; tau60 0.643")
print("reference, ColPali on same pages (E3): float 0.6954/R@1 0.375; sign 98.4%, R@1 0.375; tau 0.762 (shared-top10)")
print("subject-only floor on this corpus: R@1 = (60*0.2+12)/72 = 0.333, nDCG@5 = (60*0.59+12)/72 = 0.658")
