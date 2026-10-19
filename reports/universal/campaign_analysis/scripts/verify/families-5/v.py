import json,glob,itertools,numpy as np,os
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/"
D={}
for f in sorted(glob.glob(R+"*/wide/*_scifact.json")):
    f=f.replace(chr(92),"/"); j=json.load(open(f)); m=f.split("campaign/")[1].split("/")[0]
    D[m]={r['label']:r for r in j['rows']}
    D[m]['_b']=j['baseline']
labs=set.intersection(*[set(k for k in d if k!='_b') for d in D.values()])
labs=sorted(labs); print(len(labs),"shared labels"); print(list(D['colbert-ir--colbertv2.0']['_b'].items())[:8])
ms=list(D)
def prof(m,met): return np.array([1-D[m][l]['retention:'+met] for l in labs])
rng=np.random.default_rng(0)
for met in ['ndcg@10','ndcg@5']:
    P={m:prof(m,met) for m in ms}
    pr=[(np.sqrt(np.mean((P[a]-P[b])**2)),a,b) for a,b in itertools.combinations(ms,2)]
    pr.sort(); print(met,len(pr))
    for i,(d,a,b) in enumerate(pr):
        if 'colbertv2' in a and 'colbertv2' in b: print(' colbertv2 pair',d,'rank',i+1)
    print(' median',np.median([p[0] for p in pr]),'top5',[(round(p[0],4),p[1][:20],p[2][:20]) for p in pr[:5]])
    for k in ['LFM2','LateOn','mxbai-edge']:
        for i,(d,a,b) in enumerate(pr):
            if a.split('--')[1].startswith(k) or True:
                pass
    for i,(d,a,b) in enumerate(pr):
        sa,sb=a.split('--')[1],b.split('--')[1]
        if {sa,sb} in [{'LFM2-ColBERT-350M','LFM2.5-ColBERT-350M'},{'LateOn','mLateOn'},{'mxbai-edge-colbert-v0-17m','mxbai-edge-colbert-v0-32m'}]: print('  ',sa,sb,round(d,4),i+1)
# subset sensitivity: random 17 labels, rank of colbertv2 pair
a,b='colbert-ir--colbertv2.0','lightonai--colbertv2.0'
ranks=[]
for _ in range(500):
    idx=rng.choice(len(labs),17,replace=False)
    P={m:prof(m,'ndcg@10')[idx] for m in ms}
    pr=sorted((np.sqrt(np.mean((P[x]-P[y])**2)),x,y) for x,y in itertools.combinations(ms,2))
    ranks.append([i+1 for i,(d,x,y) in enumerate(pr) if {x,y}=={a,b}][0])
print('rank over random 17-label subsets: median',np.median(ranks),'share rank<=3',np.mean(np.array(ranks)<=3),'share rank1',np.mean(np.array(ranks)==1))
# bias check: mean loss and relation of RMSD to difference in mean loss
P={m:prof(m,'ndcg@10') for m in ms}
for m in ms: print(m[:40],round(P[m].mean(),3),D[m]['_b'].get('vectors_per_doc'),D[m]['_b'].get('ndcg@10'))
# interval overlap
n=0;lg=[]
for l in labs:
    r1,r2=D[a][l],D[b][l]
    i1,i2=r1['retention_ci:ndcg@10'],r2['retention_ci:ndcg@10']
    if i1[1]<i2[0] or i2[1]<i1[0]: n+=1
    lg.append(abs(r1['retention:ndcg@10']-r2['retention:ndcg@10']))
print('nonoverlap',n,'maxdiff',max(lg),'mean',np.mean(lg))
# CI width typical
print('mean CI width',np.mean([D[a][l]['retention_ci:ndcg@10'][1]-D[a][l]['retention_ci:ndcg@10'][0] for l in labs]))
# expected RMSD from query noise alone ~ CI half width /1.96 
