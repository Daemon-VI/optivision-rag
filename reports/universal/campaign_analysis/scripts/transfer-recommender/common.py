import json, re, math, collections
import numpy as np
from optivision.adapters import MODELS
D = json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']

# approximate parameter counts (millions), from names / public model cards; used ONLY as an optional kNN feature
PARAMS = {
 'ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1':7000,'Cognitive-Lab/ColNetraEmbed':4000,'Hcompany/NeoMME-260M-Retriever-ST-late':260,
 'Hcompany/NeoMME-800M-Retriever-ST-late':800,'LiquidAI/LFM2-ColBERT-350M':350,'LiquidAI/LFM2.5-ColBERT-350M':350,
 'Metric-AI/ColQwen2.5-3b-multilingual-v1.0':3000,'ModernVBERT/colmodernvbert':250,'NeuML/colbert-bert-tiny':4,
 'answerdotai/answerai-colbert-small-v1':33,'colbert-ir/colbertv2.0':110,'jinaai/jina-colbert-v2':560,'lightonai/ColBERT-Zero':150,
 'lightonai/GTE-ModernColBERT-v1':150,'lightonai/LateOn':150,'lightonai/Reason-ModernColBERT':150,'lightonai/colbertv2.0':110,
 'lightonai/mLateOn':300,'mixedbread-ai/mxbai-colbert-large-v1':335,'mixedbread-ai/mxbai-edge-colbert-v0-17m':17,
 'mixedbread-ai/mxbai-edge-colbert-v0-32m':32,'nomic-ai/colnomic-embed-multimodal-3b':3000,'nomic-ai/colnomic-embed-multimodal-7b':7000,
 'tsystems/colqwen2.5-3b-multilingual-v1.0':3000,'tsystems/colqwen2.5-3b-multilingual-v1.0-merged':3000,'vidore/colSmol-500M':500,
 'vidore/colpali-v1.3-merged':3000,'vidore/colqwen-omni-v0.1':3000,'vidore/colqwen2-v0.1':2200,'vidore/colqwen2-v1.0-hf':2200,
 'vidore/colqwen2-v1.0-merged':2200,'vidore/colqwen2.5-v0.1':3000,'vidore/colqwen2.5-v0.2':3000,'vidore/colsmolvlm-v0.1':256,
 'vidore/colpali':3000,'vidore/colpali-v1.1':3000,'vidore/colpali-v1.2':3000,'vidore/colpali-v1.2-hf':3000,'vidore/colpali-v1.2-merged':3000,
 'vidore/colpali-v1.3':3000,'vidore/colpali-v1.3-hf':3000,
}
def parse_size(m):
    if m in PARAMS: return PARAMS[m]
    s=m.split('/')[-1]
    g=re.search(r'(\d+(?:\.\d+)?)[bB]\b|[-_](\d+(?:\.\d+)?)[bB]',s)
    mm=re.search(r'(\d+)[mM]\b',s)
    if g: return float(g.group(1) or g.group(2))*1000
    if mm: return float(mm.group(1))
    return np.nan

# sibling clusters = same / near-identical weights re-exported (hf, merged, re-uploads)
def cluster(m):
    s=m.split('/')[-1].lower()
    if 'colbertv2.0'==s: return 'colbertv2'
    if s.startswith('colqwen2.5-3b-multilingual') or s.startswith('colqwen2.5-3b-multilingual'.replace('colqwen','colqwen')): return 'colqwen25-3b-ml'
    if s=='colqwen2.5-3b-multilingual-v1.0': return 'colqwen25-3b-ml'
    for p in ['colpali-v1.2','colpali-v1.3','colqwen2-v1.0']:
        if s.startswith(p): return p
    if 'tomoro' in s: return 'tomoro'   # 4b/8b same recipe
    return s
def cl2(m):
    s=m.split('/')[-1].lower()
    if 'colqwen2.5-3b-multilingual' in s: return 'colqwen25-3b-ml'
    return cluster(m)

def canon_row(r, dim):
    merge=None; proj=None; codec='f32'
    for s in r['pipeline']['stages']:
        st=s['stage']
        if st=='hierarchical_merge':
            merge=('hm',s['ratio']) if s.get('ratio') is not None else ('hmd',s['max_distance'])
        elif st=='adaptive_merge': merge=('ad',s['radius'],str(s.get('center')))
        elif st=='project':
            f=round(s['dim']/dim,4)
            if f<1: proj=('rand' if s.get('method')=='random' else 'pca',f)
        elif st=='float16': codec='f16'
        elif st=='int8': codec='int8'
        elif st=='int4': codec='int4m' if s.get('center')=='mean' else 'int4'
        elif st=='binary': codec='bin'
        elif st=='lloyd2': codec='lloyd2'
        else: codec=st
    return (merge,proj,codec)

def key_str(k):
    m,p,c=k
    ms='' if m is None else (f'hm{m[1]}' if m[0]=='hm' else f'{m[0]}{m[1:]}')
    ps='' if p is None else f'{p[0]}{p[1]}'
    return '+'.join(x for x in (ms,ps,c) if x)

def load():
    units=[]
    for m,v in D.items():
        info=MODELS[m]
        for ds,x in v.items():
            rows={}
            for r in x['rows']:
                k=canon_row(r,info.dimension)
                if k in rows and rows[k]['retention']!=r['retention']:
                    # duplicate canonical key (e.g. project(dim) no-op vs plain) keep first; both identical by construction
                    pass
                rows.setdefault(k,r)
            units.append(dict(model=m,ds=ds,family=info.family,modality=info.modality,dim=info.dimension,
                vpd=x['rows'][0]['vectors_per_doc'],baseline=x['baseline'],note=x.get('note',''),
                f16='float16' in x.get('note',''),size=parse_size(m),cluster=cl2(m),rows=rows,
                nq=x['n_queries'],nd=x['n_docs']))
    return units
