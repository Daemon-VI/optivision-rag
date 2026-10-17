import json, re, numpy as np, pandas as pd
from optivision.adapters import MODELS
PKG="C:/Users/Lenovo/projects/optivision-package"
OUT="C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/families"
M=json.load(open(PKG+"/src/optivision/presets/measured.json"))["models"]
SHORT={"vidore/docvqa_test_subsampled":"doc","vidore/infovqa_test_subsampled":"info","BeIR/scifact":"sci"}
# lineage groups (hand-assigned from model names / cards; exploratory)
GROUP={
 "vidore/colpali":"colpali","vidore/colpali-v1.1":"colpali","vidore/colpali-v1.2":"colpali","vidore/colpali-v1.2-hf":"colpali",
 "vidore/colpali-v1.2-merged":"colpali","vidore/colpali-v1.3":"colpali","vidore/colpali-v1.3-hf":"colpali","vidore/colpali-v1.3-merged":"colpali",
 "vidore/colqwen2-v0.1":"colqwen2","vidore/colqwen2-v1.0-hf":"colqwen2","vidore/colqwen2-v1.0-merged":"colqwen2",
 "vidore/colqwen2.5-v0.1":"colqwen2.5","vidore/colqwen2.5-v0.2":"colqwen2.5","tsystems/colqwen2.5-3b-multilingual-v1.0":"colqwen2.5",
 "tsystems/colqwen2.5-3b-multilingual-v1.0-merged":"colqwen2.5","Metric-AI/ColQwen2.5-3b-multilingual-v1.0":"colqwen2.5",
 "nomic-ai/colnomic-embed-multimodal-3b":"colqwen2.5","nomic-ai/colnomic-embed-multimodal-7b":"colqwen2.5","ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1":"colqwen2.5",
 "vidore/colqwen-omni-v0.1":"colqwen-omni",
 "vidore/colSmol-500M":"smolvlm","vidore/colsmolvlm-v0.1":"smolvlm","ModernVBERT/colmodernvbert":"modernvbert",
 "Cognitive-Lab/ColNetraEmbed":"gemma-netra",
 "Hcompany/NeoMME-260M-Retriever-ST-late":"neomme","Hcompany/NeoMME-800M-Retriever-ST-late":"neomme",
 "TomoroAI/tomoro-colqwen3-embed-4b":"qwen3vl","TomoroAI/tomoro-colqwen3-embed-8b":"qwen3vl",
 "VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1":"qwen3vl","VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1":"qwen3vl",
 "nvidia/nemotron-colembed-vl-4b-v2":"qwen3vl","nvidia/nemotron-colembed-vl-8b-v2":"qwen3vl",
 "webAI-Official/webAI-ColVec1-4b":"qwen3vl","webAI-Official/webAI-ColVec1.1-8b":"qwen3vl",
 "athrael-soju/colqwen3.5-4.5B-v3":"qwen3.5","vultr/VultronRetrieverCore-Qwen3.5-4.5B":"qwen3.5","vultr/VultronRetrieverFlash-Qwen3.5-0.8B":"qwen3.5",
 "vultr/VultronRetrieverPrime-Qwen3.5-8B":"qwen3.5","tencent/EVIE-4.5B":"qwen3.5","tencent/EVIE-8B":"qwen3.5",
}
SIB={ # named sibling sets from the research question
 "colpali":["vidore/colpali-v1.1","vidore/colpali-v1.2","vidore/colpali-v1.2-hf","vidore/colpali-v1.2-merged","vidore/colpali-v1.3","vidore/colpali-v1.3-hf","vidore/colpali-v1.3-merged"],
 "colqwen2":["vidore/colqwen2-v0.1","vidore/colqwen2-v1.0-hf","vidore/colqwen2-v1.0-merged"],
 "nemotron":["nvidia/nemotron-colembed-vl-4b-v2","nvidia/nemotron-colembed-vl-8b-v2"],
 "evie":["tencent/EVIE-4.5B","tencent/EVIE-8B"],
 "vultron":["vultr/VultronRetrieverFlash-Qwen3.5-0.8B","vultr/VultronRetrieverCore-Qwen3.5-4.5B","vultr/VultronRetrieverPrime-Qwen3.5-8B"],
 "sauerkraut":["VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1","VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1"],
 "mxbai-edge":["mixedbread-ai/mxbai-edge-colbert-v0-17m","mixedbread-ai/mxbai-edge-colbert-v0-32m"],
 "neomme":["Hcompany/NeoMME-260M-Retriever-ST-late","Hcompany/NeoMME-800M-Retriever-ST-late"],
 "colbertv2":["colbert-ir/colbertv2.0","lightonai/colbertv2.0"],
 "colqwen2.5-3b-tsys":["tsystems/colqwen2.5-3b-multilingual-v1.0","tsystems/colqwen2.5-3b-multilingual-v1.0-merged"],
}
def info(m):
    mi=MODELS.get(m)
    ev=(mi.evidence if mi else "") or ""
    note=""
    for ds,x in M[m].items(): note=x.get("note") or note
    f16 = ("float16 inference" in ev) or ("float16 inference" in note)
    return dict(model=m, family=(mi.family if mi else None), modality=(mi.modality if mi else None),
        dim=(mi.dimension if mi else None), f16=f16, group=GROUP.get(m,"text" if (mi and mi.modality=="text") else "other"))
def size_b(m):
    # parameter count in billions, hand-read from model names / cards (approximate)
    T={"vidore/colpali":3,"vidore/colqwen2-v0.1":2.2,"vidore/colqwen2.5-v0.1":3,"vidore/colSmol-500M":0.5,"vidore/colsmolvlm-v0.1":0.256,
       "tencent/EVIE-4.5B":4.5,"tencent/EVIE-8B":8,"vultr/VultronRetrieverFlash-Qwen3.5-0.8B":0.8,"vultr/VultronRetrieverCore-Qwen3.5-4.5B":4.5,
       "vultr/VultronRetrieverPrime-Qwen3.5-8B":8,"VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1":2,"VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1":4,
       "nvidia/nemotron-colembed-vl-4b-v2":4,"nvidia/nemotron-colembed-vl-8b-v2":8,"Hcompany/NeoMME-260M-Retriever-ST-late":0.26,"Hcompany/NeoMME-800M-Retriever-ST-late":0.8,
       "mixedbread-ai/mxbai-edge-colbert-v0-17m":0.017,"mixedbread-ai/mxbai-edge-colbert-v0-32m":0.032}
    return T.get(m)
def long_df():
    rows=[]
    for m,v in M.items():
        for ds,x in v.items():
            for r in x["rows"]:
                lo,hi=r["retention_interval"]
                rows.append(dict(model=m,ds=SHORT[ds],label=r["label"],ret=r["retention"],lo=lo,hi=hi,comp=r["compression"],vpd=r["vectors_per_doc"],bpd=r["bytes_per_doc"],base=x["baseline"],nq=x["n_queries"]))
    return pd.DataFrame(rows)
def mat(df, labels=None, ds=None):
    d=df if ds is None else df[df.ds==ds]
    P=d.pivot_table(index=["model","ds"],columns="label",values="ret")
    if labels is not None: P=P[labels]
    return P

def labelsA(df):
    P=mat(df); n=P.notna().sum()
    return [l for l in P.columns if n[l]>=88 and 'project' not in l and l!='float32']
CODEC4={'int8(per_vector)':'none > int8/vec','int4':'none > int4','binary':'none > binary','float16':'none > float16'}
def se_mat(df,labels):
    d=df.copy(); d['se']=(d.hi-d.lo)/3.92
    return d.pivot_table(index=["model","ds"],columns="label",values="se")[labels]

GROUP2=dict(GROUP)
for m in ["lightonai/ColBERT-Zero","lightonai/GTE-ModernColBERT-v1","lightonai/LateOn","lightonai/mLateOn","lightonai/Reason-ModernColBERT"]: GROUP2[m]="modernbert-colbert"
for m in SIB["colbertv2"]: GROUP2[m]="colbertv2"
for m in SIB["mxbai-edge"]: GROUP2[m]="mxbai-edge"
for m in ["LiquidAI/LFM2-ColBERT-350M","LiquidAI/LFM2.5-ColBERT-350M"]: GROUP2[m]="lfm2"
def L1labels(df):
    A=labelsA(df)
    return [l for l in A if l!='float16' and not l.endswith('> float16') and 'int4' not in l or l=='int4(mean)' or l.endswith('> int4(mean)')]
def L1b(df):  # drop plain int4 (collapse) and float16
    return [l for l in labelsA(df) if l!='float16' and not l.endswith('> float16') and not (l=='int4' or l.endswith('> int4'))]

TWIN={ # underlying checkpoint key (identical weights re-packaged)
 "vidore/colpali-v1.2":"cp12","vidore/colpali-v1.2-hf":"cp12","vidore/colpali-v1.2-merged":"cp12",
 "vidore/colpali-v1.3":"cp13","vidore/colpali-v1.3-hf":"cp13","vidore/colpali-v1.3-merged":"cp13",
 "vidore/colqwen2-v1.0-hf":"cq2","vidore/colqwen2-v1.0-merged":"cq2",
 "tsystems/colqwen2.5-3b-multilingual-v1.0":"ts","tsystems/colqwen2.5-3b-multilingual-v1.0-merged":"ts",
 "colbert-ir/colbertv2.0":"cb2","lightonai/colbertv2.0":"cb2",
}
def twin(m): return TWIN.get(m,m)
