import json, numpy as np, re
from scipy import stats
from optivision.adapters import MODELS
d=json.load(open('src/optivision/presets/measured.json'))['models']
reg=dict(MODELS)
rng=np.random.default_rng(0)
def f16(k):
    e=reg[k].evidence if k in reg else ''
    return 'float16 inference' in e
def prof(k,ds):
    return {r['label']:(1-r['retention'])*100 for r in d[k][ds]['rows']}
DS=['vidore/docvqa_test_subsampled','vidore/infovqa_test_subsampled']
TRIV={'float32','float16'}
def pair(a,b,datasets=DS,only=None,excl=TRIV):
    res=[]
    for ds in datasets:
        if ds not in d[a] or ds not in d[b]: continue
        pa,pb=prof(a,ds),prof(b,ds)
        L=[l for l in pa if l in pb and l not in excl and (only is None or only(l))]
        res.append(np.mean([pb[l]-pa[l] for l in L]))
    return np.mean(res), res, len(L)
pairs=[('nvidia/nemotron-colembed-vl-4b-v2','nvidia/nemotron-colembed-vl-8b-v2'),
('tencent/EVIE-4.5B','tencent/EVIE-8B'),
('vultr/VultronRetrieverFlash-Qwen3.5-0.8B','vultr/VultronRetrieverCore-Qwen3.5-4.5B'),
('vultr/VultronRetrieverCore-Qwen3.5-4.5B','vultr/VultronRetrieverPrime-Qwen3.5-8B'),
('VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1','VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1'),
('TomoroAI/tomoro-colqwen3-embed-4b','TomoroAI/tomoro-colqwen3-embed-8b'),
('nomic-ai/colnomic-embed-multimodal-3b','nomic-ai/colnomic-embed-multimodal-7b'),
('Hcompany/NeoMME-260M-Retriever-ST-late','Hcompany/NeoMME-800M-Retriever-ST-late'),
('mixedbread-ai/mxbai-edge-colbert-v0-17m','mixedbread-ai/mxbai-edge-colbert-v0-32m')]
for nm,only in [('all-nontrivial',None),('binary-only',lambda l:l=='binary'),('codec-only',lambda l:'merge' not in l and 'project' not in l),('merge-only',lambda l:'merge' in l and 'project' not in l and '>' not in l)]:
    print('==',nm); v=[]
    for a,b in pairs:
        ds=DS if 'mxbai' not in a else ['scifact']
        sci=[k for k in d[a]]
        ds=[x for x in d[a]]
        m,r,n=pair(a,b,ds,only)
        v.append(m)
        print(f"{a.split('/')[1][:30]:30s}->{b.split('/')[1][-12:]:12s} f16:{f16(a)!s:5}{f16(b)!s:5} mean {m:+.2f} per-ds {np.round(r,2)} nlab {n}")
    v=np.array(v[:8]); 
    bs=[rng.choice(v,len(v)).mean() for _ in range(5000)]
    print('image 8 pairs mean %.3f CI %.2f..%.2f worse(>0) %d/8'%(v.mean(),*np.percentile(bs,[2.5,97.5]),(v>0).sum()), 'sign p',stats.binomtest(int((v>0).sum()),8).pvalue)
print({k:(f16(k)) for k in d if k in reg and reg[k].modality=='image' and f16(k)})
print([ (k) for k in d if k not in reg])
