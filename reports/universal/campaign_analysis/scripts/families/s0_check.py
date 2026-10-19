from common import *
import glob, json, re
rows=[]
for f in glob.glob(PKG+'/reports/universal/campaign/*/wide/*_*.json'):
    f=f.replace(chr(92),'/')
    j=json.load(open(f))
    if 'baseline' not in j or 'rows' not in j: continue
    folder=f.split('campaign/')[1].split('/')[0]
    m=folder.replace('--','/',1)
    ds={'docvqa_test_subsampled':'vidore/docvqa_test_subsampled','infovqa_test_subsampled':'vidore/infovqa_test_subsampled','scifact':'BeIR/scifact'}
    key=[k for k in ds if f.endswith(k+'.json')]
    if not key: continue
    b=j['baseline']
    if m in M and ds[key[0]] in M[m]:
        mb=M[m][ds[key[0]]]['baseline']
        rows.append((m,ds[key[0]],mb,b['ndcg@5'],b['ndcg@10']))
R=pd.DataFrame(rows,columns=['model','ds','measured_baseline','wide_ndcg5','wide_ndcg10'])
R['is5']=(R.measured_baseline-R.wide_ndcg5).abs()<6e-5; R['is10']=(R.measured_baseline-R.wide_ndcg10).abs()<6e-5
print(len(R)); print(R.groupby(R.ds)[['is5','is10']].agg(['sum','count']))
print(R[~R.is5 & ~R.is10])
# compare retention in measured.json to wide rows
bad=0;tot=0
for f in glob.glob(PKG+'/reports/universal/campaign/*/wide/*_*.json'):
    f=f.replace(chr(92),'/'); j=json.load(open(f))
    if 'rows' not in j: continue
    folder=f.split('campaign/')[1].split('/')[0]; m=folder.replace('--','/',1)
    for k,v in {'docvqa_test_subsampled':'vidore/docvqa_test_subsampled','infovqa_test_subsampled':'vidore/infovqa_test_subsampled','scifact':'BeIR/scifact'}.items():
        if f.endswith(k+'.json') and m in M and v in M[m]:
            mr={r['label']:r['retention'] for r in M[m][v]['rows']}
            for r in j['rows']:
                if r['label'] in mr:
                    tot+=1
                    r5=r.get('retention:ndcg@5'); r10=r.get('retention:ndcg@10')
                    if abs(mr[r['label']]-r5)>6e-5: bad+=1
print('rows compared',tot,'retention != retention:ndcg@5 :',bad)
import collections
c=collections.Counter()
for f in glob.glob(PKG+'/reports/universal/campaign/*/wide/*_*.json'):
    f=f.replace(chr(92),'/'); j=json.load(open(f))
    if 'rows' not in j: continue
    folder=f.split('campaign/')[1].split('/')[0]; m=folder.replace('--','/',1)
    for k,v in {'docvqa_test_subsampled':'vidore/docvqa_test_subsampled','infovqa_test_subsampled':'vidore/infovqa_test_subsampled','scifact':'BeIR/scifact'}.items():
        if f.endswith(k+'.json') and m in M and v in M[m]:
            mr={r['label']:r['retention'] for r in M[m][v]['rows']}
            for r in j['rows']:
                if r['label'] in mr:
                    c[(v,'m=r5' if abs(mr[r['label']]-r['retention:ndcg@5'])<6e-5 else ('m=r10' if abs(mr[r['label']]-r['retention:ndcg@10'])<6e-5 else 'neither'))]+=1
print(c)
