import os

"""Replay the original 9-variant table through the new stages and compare with the old bench JSON."""
import json
import time
from pathlib import Path

from optivision.benchmark import legacy_variants, load_legacy_dataset, run_matrix, save_result

D = os.environ.get("OPTIVISION_DATA", "data")
S = os.environ.get("OPTIVISION_OUT", "reports/universal/raw")
jobs = {
    "E1-colsmol-generated": (D + "/cache/colsmol.npz", D + "/corpus/queries.json", S + "/phase0/e1_colsmol/benchmark.json"),
    "E3-colpali-generated": (D + "/cache/colpali_generated.npz", D + "/corpus/queries.json", S + "/phase0/e3_colpali_generated/benchmark.json"),
    "E2-colpali-infovqa": (D + "/cache/colpali_infovqa_test_subsampled.npz", D + "/vidore_infovqa/queries.json", S + "/phase0/e2_colpali_infovqa/benchmark.json"),
    "E2-colpali-docvqa": (D + "/cache/colpali_docvqa_test_subsampled.npz", D + "/vidore_docvqa_test_subsampled/queries.json", S + "/phase0/e2_colpali_docvqa/benchmark.json"),
}
for name, (cache, qj, old_json) in jobs.items():
    t = time.time()
    ds = load_legacy_dataset(cache, qj, name=name)
    variants = legacy_variants(); variants.pop("baseline-float32")
    res = run_matrix(ds, variants, metrics=["ndcg@5", "ndcg@10", "recall@1", "mrr@10"], n_boot=0)
    save_result(res, S + "/phase2/legacy_equiv", name)
    old = {r["variant"]: r for r in json.loads(Path(old_json).read_text(encoding="utf-8"))["rows"]}
    worst = 0.0
    for r in res["rows"]:
        o = old[r["label"]]
        d5 = abs(r["ndcg@5"] - o["ndcg@5"]); d10 = abs(r["ndcg@10"] - o["ndcg@10"])
        dkb = abs(r["bytes_per_doc"] / 1e3 - o["kb_per_page"]); dvec = abs(r["vectors_per_doc"] - o["tokens_per_page"])
        worst = max(worst, d5, d10)
        flag = "" if max(d5, d10) < 1e-9 and dkb < 1e-6 and dvec < 1e-9 else "   <-- differs"
        print(f"{name:22s} {r['label']:22s} nDCG@5 new {r['ndcg@5']:.4f} old {o['ndcg@5']:.4f} | nDCG@10 {r['ndcg@10']:.4f}/{o['ndcg@10']:.4f} | KB {r['bytes_per_doc']/1e3:.2f}/{o['kb_per_page']:.2f} | vec {r['vectors_per_doc']:.1f}/{o['tokens_per_page']:.1f}{flag}", flush=True)
    print(f"{name}: max |delta nDCG| = {worst:.2e}; {time.time()-t:.0f}s", flush=True)
