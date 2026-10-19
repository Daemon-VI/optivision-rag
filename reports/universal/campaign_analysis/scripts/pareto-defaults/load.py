import json,glob,os,sys
import numpy as np,pandas as pd
REPO="C:/Users/Lenovo/projects/optivision-package"
sys.path.insert(0,REPO+"/src")
M=json.load(open(REPO+"/src/optivision/presets/measured.json"))["models"]
def measured():
    rec=[]
    for m,ds in M.items():
        for d,v in ds.items():
            for r in v["rows"]:
                rec.append(dict(model=m,dataset=d,label=r["label"],ret=r["retention"],lo=r["retention_interval"][0],hi=r["retention_interval"][1],comp=r["compression"],bpd=r["bytes_per_doc"],vpd=r["vectors_per_doc"],base=v["baseline"],nd=v["n_docs"],nq=v["n_queries"],stages=len(r["pipeline"]["stages"])))
    return pd.DataFrame(rec)
def wide():
    rec=[];tier=[];base=[]
    for f in glob.glob(REPO+"/reports/universal/campaign/*/wide/*.json"):
        d=json.load(open(f)); m=f.replace("\\","/").split("/campaign/")[1].split("/")[0].replace("--","/",1)
        b=d["baseline"]; base.append(dict(model=m,dataset=d["dataset"],dtype=b["encoder"].get("dtype"),vpd=b["vectors_per_doc"],dim=b["dim"],scan=b["scan_ms_per_query"],nd=d["n_docs"],nq=d["n_queries"],f32=b["float32_bytes_per_doc"],ndcg=b["ndcg@5"]))
        for r in d["rows"]:
            rec.append(dict(model=m,dataset=d["dataset"],group=r["group"],label=r["label"],comp=r["compression_vs_float32"],bpd=r["bytes_per_doc"],vpd=r["vectors_per_doc"],scan=r["scan_ms_per_query"],ret=r["retention:ndcg@5"],lo=r["retention_ci:ndcg@5"][0],hi=r["retention_ci:ndcg@5"][1],ret1=r["retention:recall@1"],pareto=r["pareto"],cs=r["compress_seconds"]))
        for t in d["tiered"]:
            tier.append(dict(model=m,dataset=d["dataset"],hot=t["hot"],cold=t["cold"],cand=t["candidates"],ram=t["ram_bytes_per_doc"],disk=t["disk_bytes_per_doc"],ramc=t["ram_compression_vs_float32"],diskc=t["disk_compression_vs_float32"],ret=t["retention:ndcg@5"],lo=t["retention_ci:ndcg@5"][0],hi=t["retention_ci:ndcg@5"][1],hotms=t["hot_ms_per_query"],resms=t["rescore_ms_per_query"]))
    return pd.DataFrame(rec),pd.DataFrame(tier),pd.DataFrame(base)

def wide_all():
    import glob
    rec=[];tier=[];base=[]
    files=glob.glob(REPO+"/reports/universal/campaign/*/wide/*.json")+[f for f in glob.glob(REPO+"/reports/universal/wide/*_subsampled.json")]
    for f in files:
        f=f.replace("\\","/"); d=json.load(open(f))
        if "campaign" in f: m=f.split("/campaign/")[1].split("/")[0].replace("--","/",1)
        else: m=d["baseline"]["encoder"]["model"]
        ds=[x for x in ("docvqa","infovqa","scifact") if x in f.lower() or x in d["dataset"].lower()][0]
        b=d["baseline"]; key=m+"|"+ds
        base.append(dict(key=key,model=m,ds=ds,dtype=str(b["encoder"].get("dtype")),vpd=b["vectors_per_doc"],dim=b["dim"],scan=b["scan_ms_per_query"],f32=b["float32_bytes_per_doc"],ndcg=b["ndcg@5"],nd=d["n_docs"],nq=d["n_queries"],src="campaign" if "campaign" in f else "older"))
        for r in d["rows"]:
            rec.append(dict(key=key,model=m,ds=ds,group=r["group"],label=r["label"],comp=r["compression_vs_float32"],bpd=r["bytes_per_doc"],vpd=r["vectors_per_doc"],dim=r["dim"],scan=r["scan_ms_per_query"],ret=r["retention:ndcg@5"],lo=r["retention_ci:ndcg@5"][0],hi=r["retention_ci:ndcg@5"][1],pareto=r["pareto"],cs=r["compress_seconds"],bits=r["bits_per_dim"]))
        for t in d["tiered"]:
            tier.append(dict(key=key,model=m,ds=ds,hot=t["hot"],cold=t["cold"],cand=t["candidates"],ram=t["ram_bytes_per_doc"],disk=t["disk_bytes_per_doc"],ramc=t["ram_compression_vs_float32"],diskc=t["disk_compression_vs_float32"],ret=t["retention:ndcg@5"],lo=t["retention_ci:ndcg@5"][0],hi=t["retention_ci:ndcg@5"][1],hotms=t["hot_ms_per_query"],resms=t["rescore_ms_per_query"]))
    return pd.DataFrame(rec),pd.DataFrame(tier),pd.DataFrame(base)
