import json,glob,numpy as np
from scipy.stats import mannwhitneyu
from optivision.adapters import MODELS
m=json.load(open('src/optivision/presets/measured.json'))['models']
def get(lab):
    return {mid:np.mean([r['retention'] for d in ds.values() for r in d['rows'] if r['label']==lab]) for mid,ds in m.items() if any(r['label']==lab for d in ds.values() for r in d['rows'])}
R=get('project(32, random)');P25=get('hierarchical_merge(0.25) > project(32)');P33=get('hierarchical_merge(0.33) > project(32)')
M25=get('hierarchical_merge(0.25)');M33=get('hierarchical_merge(0.33)')
mod=lambda k:MODELS[k].modality
rng=np.random.default_rng(1)
def rep(name,P,Mr=None):
    gaps={}
    for k in R:
        if k in P: gaps[k]=P[k]-R[k]
    gt=np.array([v for k,v in gaps.items() if mod(k)=='text']);gi=np.array([v for k,v in gaps.items() if mod(k)=='image'])
    b=[rng.choice(gt,len(gt)).mean()-rng.choice(gi,len(gi)).mean() for _ in range(5000)]
    print(name,'n',len(gt),len(gi),'pca text',np.mean([P[k] for k in gaps if mod(k)=='text']).round(3),'img',np.mean([P[k] for k in gaps if mod(k)=='image']).round(3),'gap',gt.mean().round(3),gi.mean().round(3),'diff CI',np.percentile(b,[2.5,97.5]).round(3),'MW p',mannwhitneyu(gt,gi,alternative='greater').pvalue.round(4))
rep('merge.25',P25);rep('merge.33',P33)
# PCA efficiency: pca/merge-only
for nm,P,M in(('.25',P25,M25),('.33',P33,M33)):
    for mm in('text','image'):
        ks=[k for k in P if mod(k)==mm]
        print(nm,mm,'merge only',np.mean([M[k] for k in ks]).round(3),'merge+pca32',np.mean([P[k] for k in ks]).round(3))
txt=[k for k in R if mod(k)=='text']
for k in sorted(txt,key=lambda k:R[k]): print(k,round(R[k],3),round(P25[k],3),round(P33[k],3))
img=sorted((round(R[k],3),k) for k in R if mod(k)=='image');print(img[:8],img[-3:])
# robustness: drop 4 collapsed -> text mean; text rnd quantile
rest=[R[k] for k in txt if R[k]>0.05];print('text rnd excl 4',np.mean(rest).round(3),len(rest))
# image per dataset
for ds in('docvqa','infovqa'): pass
# fraction of text models with rnd<0.5 vs image
print(sum(R[k]<0.5 for k in txt),len(txt),sum(R[k]<0.5 for k in R if mod(k)=='image'),'img')
# per-dataset image values
for k in R:
    if mod(k)=='image':
        pass
import collections
dsr=collections.defaultdict(list)
for k,ds in m.items():
    for d,x in ds.items():
        for r in x['rows']:
            if r['label']=='project(32, random)': dsr[(mod(k),d)].append(r['retention'])
print({kk:(len(v),round(np.mean(v),3)) for kk,v in dsr.items()})
# model families for text
print({k:(MODELS[k].family,MODELS[k].dimension, [ (d,x['baseline']) for d,x in m[k].items()][:1]) for k in txt})
print('--- leave-out')
from scipy.stats import mannwhitneyu
drop={'lightonai/LateOn','lightonai/mLateOn','lightonai/GTE-ModernColBERT-v1','LiquidAI/LFM2-ColBERT-350M','LiquidAI/LFM2.5-ColBERT-350M','lightonai/ColBERT-Zero','lightonai/Reason-ModernColBERT'}
for nm,ex in(('drop4',{'lightonai/LateOn','lightonai/mLateOn','lightonai/GTE-ModernColBERT-v1','LiquidAI/LFM2-ColBERT-350M'}),('drop7 lighton/LFM',drop)):
    ks=[k for k in txt if k not in ex]
    gt=np.array([P33[k]-R[k] for k in ks]);gi=np.array([P33[k]-R[k] for k in R if mod(k)=='image'])
    b=[rng.choice(gt,len(gt)).mean()-rng.choice(gi,len(gi)).mean() for _ in range(5000)]
    print(nm,len(ks),'rnd',np.mean([R[k] for k in ks]).round(3),'gapdiff',(gt.mean()-gi.mean()).round(3),np.percentile(b,[2.5,97.5]).round(3),mannwhitneyu(gt,gi,alternative='greater').pvalue.round(4))
