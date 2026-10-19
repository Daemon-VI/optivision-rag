import json,glob,os,re,numpy as np,collections
R='C:/Users/Lenovo/projects/optivision-package/'
C=R+'reports/universal/campaign/'
M=json.load(open(R+'src/optivision/presets/measured.json'))['models']
print(len(M), sum(len(d['rows']) for m in M.values() for d in m.values()))
# which models' datasets
kinds=collections.Counter()
wide={}
for f in [x.replace(chr(92),'/') for x in glob.glob(C+'*/wide/*.json')]:
    org=f.split('/')[-3] if '/' in f else f
    d=json.load(open(f)); wide[f.replace(chr(92),'/')]=d
print('wide',len(wide))
runs={}
for f in glob.glob(C+'*/run.json'):
    d=json.load(open(f)); runs[d['model']]=d
print('runs',len(runs), collections.Counter(r.get('kind') for r in runs.values()))
# baseline check
txt=0;img=0;bad=[]
for f,d in wide.items():
    ds=d['dataset']; b=d['baseline']
    run=runs.get(os.path.dirname(os.path.dirname(f)).split('/')[-1].replace('--','/',1))
    kind='text' if 'scifact' in ds else 'image'
    # find model in M
    mid=os.path.dirname(os.path.dirname(f)).split('/')[-1].replace('--','/',1)
    if mid not in M: bad.append(('nomodel',mid));continue
    for k,v in M[mid].items():
        if (kind=='text')==('scifact' in k.lower()):
            e5=abs(v['baseline']-b['ndcg@5'])<1e-9; e10=abs(v['baseline']-b['ndcg@10'])<1e-9
            if kind=='text': txt+=1; 
            else: img+=1
            bad.append((kind,mid,k,e5,e10))
print(collections.Counter((k,e5,e10) for k,_,_,e5,e10 in [b for b in bad if len(b)==5]))
print([b for b in bad if len(b)==2][:5])
# text rows retention comparison
diffs=[];ch99=0;pass99=0;ch95=0;n95=0
for f,d in wide.items():
    if 'scifact' not in d['dataset']: continue
    for r in d['rows']:
        a=r['retention:ndcg@5'];b=r['retention:ndcg@10']
        diffs.append(b-a)
        p5,p10=a>=.99,b>=.99
        if p5 or p10:
            pass99+=1; ch99+= p5!=p10
        q5,q10=a>=.95,b>=.95
        n95+=1; ch95+= q5!=q10
print('text rows',len(diffs),np.mean(diffs),min(diffs),max(diffs),'0.99 pass-either',pass99,ch99,'0.95 any-either',sum(1 for _ in [0]),ch95,n95)
# 0.95 "of 340" = either passes
c=0;t=0
for f,d in wide.items():
    if 'scifact' not in d['dataset']: continue
    for r in d['rows']:
        a=r['retention:ndcg@5'];b=r['retention:ndcg@10']
        if a>=.95 or b>=.95:
            t+=1;c+=(a>=.95)!=(b>=.95)
print('0.95',c,t)
# compare measured.json retention to wide for text
mm=0
for f,d in wide.items():
    mid=os.path.dirname(os.path.dirname(f)).split('/')[-1].replace('--','/',1)
    if mid not in M: continue
    for k,v in M[mid].items():
        if ('scifact' in k.lower())!=('scifact' in d['dataset']): continue
        lab={r['label']:r for r in d['rows']}
        for r in v['rows']:
            w=lab.get(r['label'])
            if w is None: continue
            m5=abs(r['retention']-w['retention:ndcg@5'])<1e-9;m10=abs(r['retention']-w['retention:ndcg@10'])<1e-9
            kk=('text' if 'scifact' in d['dataset'] else 'img',m5,m10)
            kinds[kk]+=1
print(kinds)
# unit norm
fp=collections.defaultdict(list)
for mid,r in runs.items():
    u=r.get('check_retriever',{}).get('unit_norm_max_error')
    sd=r.get('spec',{}).get('dtype')
    fp[(r.get('kind'),sd)].append((mid,u))
for k,v in sorted(fp.items(),key=str): print(k,[(m.split('/')[-1],'%.1e'%u if u is not None else None) for m,u in v])
