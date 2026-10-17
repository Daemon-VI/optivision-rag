exec(open(__file__.replace('v3.py','v.py')).read().split("for nm,only in")[0])
img=[k for k in d if reg[k].modality=='image' and 'binary' in prof(k,DS[0])]
print(len(img),'image models')
size={'ApsaraStackMaaS/EvoQwen2.5-VL-Retriever-7B-v1':7,'Cognitive-Lab/ColNetraEmbed':4,'Hcompany/NeoMME-260M-Retriever-ST-late':.26,'Hcompany/NeoMME-800M-Retriever-ST-late':.8,
'Metric-AI/ColQwen2.5-3b-multilingual-v1.0':3,'TomoroAI/tomoro-colqwen3-embed-4b':4,'TomoroAI/tomoro-colqwen3-embed-8b':8,'VAGOsolutions/SauerkrautLM-ColQwen3-2b-v0.1':2,'VAGOsolutions/SauerkrautLM-ColQwen3-4b-v0.1':4,
'nomic-ai/colnomic-embed-multimodal-3b':3,'nomic-ai/colnomic-embed-multimodal-7b':7,'nvidia/nemotron-colembed-vl-4b-v2':4,'nvidia/nemotron-colembed-vl-8b-v2':8,'tencent/EVIE-4.5B':4.5,'tencent/EVIE-8B':8,
'tsystems/colqwen2.5-3b-multilingual-v1.0':3,'vidore/colSmol-500M':.5,'vultr/VultronRetrieverCore-Qwen3.5-4.5B':4.5,'vultr/VultronRetrieverFlash-Qwen3.5-0.8B':.8,'vultr/VultronRetrieverPrime-Qwen3.5-8B':8,
'webAI-Official/webAI-ColVec1-4b':4,'webAI-Official/webAI-ColVec1.1-8b':8,'athrael-soju/colqwen3.5-4.5B-v3':4.5}
for ds in DS:
  allp={k:prof(k,ds) for k in img if ds in d[k]}
  common=set.intersection(*[set(p) for p in allp.values()])-TRIV
  for nm,flt in [('noproject',lambda l:'project' not in l),('all common',lambda l:True)]:
    L=sorted(l for l in common if flt(l))
    ks=[k for k in size if k in allp]
    x=[np.log(size[k]) for k in ks]; y=[np.mean([allp[k][l] for l in L]) for k in ks]
    r=stats.spearmanr(x,y)
    # drop sibling dupes: keep one per family? leave-one-out of extreme
    print(ds[6:12],nm,len(L),'labels',len(ks),'models rho %.2f p %.3f'%(r.statistic,r.pvalue))
    # excluding Vultron 0.8B (extreme)
    ks2=[k for k in ks if 'Flash' not in k]
    r2=stats.spearmanr([np.log(size[k]) for k in ks2],[np.mean([allp[k][l] for l in L]) for k in ks2]); print('   w/o Flash %.2f p %.3f'%(r2.statistic,r2.pvalue))
# FE regression
def reg_fe(flt,model_f16=True):
    rows=[]
    for k in img:
        for ds in DS:
            if ds in d[k]: rows.append((k,ds))
    allp={(k,ds):prof(k,ds) for k,ds in rows}
    common=set.intersection(*[set(p) for p in allp.values()])-TRIV
    L=sorted(l for l in common if flt(l))
    y=np.array([np.mean([allp[r][l] for l in L]) for r in rows])
    return rows,y,len(L)
def fit(rows,y,models):
    fams=sorted({reg[k].family for k in models})
    X=[]
    for (k,ds) in rows:
        X.append([1.0*(reg[k].family==f) for f in fams]+[1.0*(ds==DS[1]), 1.0*f16(k)])
    X=np.array(X); b=np.linalg.lstsq(X,y,rcond=None)[0]; return b[-1]
for nm,flt in [('noproject',lambda l:'project' not in l),('merge-only',lambda l:'merge' in l and 'project' not in l and '>' not in l),('codec-only',lambda l:'merge' not in l and 'project' not in l),('with project',lambda l:True)]:
    rows,y,nl=reg_fe(flt)
    models=sorted({r[0] for r in rows})
    b=fit(rows,y,models)
    bs=[]
    for _ in range(2000):
        s=rng.choice(len(models),len(models))
        rr=[];yy=[]
        for i,j in enumerate(s):
            for t,(k,ds) in enumerate(rows):
                if k==models[j]: rr.append((k,ds)); yy.append(y[t])
        # need cluster dup: families present ok
        try: bs.append(fit(rr,np.array(yy),[models[j] for j in s]))
        except Exception: pass
    print(nm,nl,'labels',len(rows),'profiles',len(models),'models f16 coef %.3f CI %.2f..%.2f'%(b,*np.percentile(bs,[2.5,97.5])))
