import json,re,numpy as np,pandas as pd
from scipy.stats import spearmanr,rankdata
rng=np.random.default_rng(0)
M=json.load(open("C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json"))["models"]
recs=[]
for mid,dss in M.items():
    for ds,d in dss.items():
        for r in d['rows']:
            recs.append(dict(model=mid,ds=ds,label=r['label'],ret=r['retention'],lo=r['retention_interval'][0],hi=r['retention_interval'][1],comp=r['compression']))
df=pd.DataFrame(recs)
print(len(df),df.model.nunique())
# canonical: stage labels split on ' > '; keep rows whose every stage is a plain label present as a single row
idx={(r.model,r.ds,r.label):r for r in df.itertuples()}
bad=re.compile(r'ward|adaptive|none|\[|random|int8/vec')
out=[]
for r in df.itertuples():
    st=r.label.split(' > ')
    if len(st)<2 or bad.search(r.label): continue
    parts=[idx.get((r.model,r.ds,s)) for s in st]
    if any(p is None for p in parts): continue
    out.append(dict(model=r.model,ds=r.ds,label=r.label,act=r.ret,alo=r.lo,ahi=r.hi,comp=r.comp,pred=np.prod([p.ret for p in parts]),
      predmin=min(p.ret for p in parts),first=parts[0].ret,nst=len(st)))
A=pd.DataFrame(out)
print('n',len(A),'models',A.model.nunique(),'units',A.groupby(['model','ds']).ngroups)
def auc(s,y):
    y=np.asarray(y,bool);s=np.asarray(s)
    if y.sum()==0 or (~y).sum()==0:return np.nan
    r=rankdata(s);return (r[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum())
def conf(g,T,mar=0):
    pos=g.act>=T;sel=g.pred>=T+mar
    tp=(pos&sel).sum();fp=(~pos&sel).sum();fn=(pos&~sel).sum()
    return tp,fp,fn
res={}
for T in [0.95,0.98,0.99,0.90]:
    tp,fp,fn=conf(A,T);pos=(A.act>=T)
    print(f"T={T} base={pos.mean():.3f} prec={tp/(tp+fp):.3f} rec={tp/(tp+fn):.3f} fp={fp} fn={fn} AUC pred={auc(A.pred,pos):.3f} min={auc(A.predmin,pos):.3f} first={auc(A['first'],pos):.3f}")
    # straddle
    mis=A[((A.pred>=T)!=(A.act>=T))]
    st=((mis.alo<=T)&(mis.ahi>=T)).mean()
    fpd=A[(A.pred>=T)&(A.act<T)]
    print('   misclass',len(mis),'straddle',round(st,3),'FP shortfall mean %.4f max %.4f'%((T-fpd.act).mean(),(T-fpd.act).max()))
tp,fp,fn=conf(A,0.98,0.01);print('0.98+.01 prec',tp/(tp+fp),'rec',tp/(tp+fn))
# cluster bootstrap over models
models=A.model.unique()
def stat(sub,T):
    tp,fp,fn=conf(sub,T);return tp/(tp+fp),tp/(tp+fn),auc(sub.pred,sub.act>=T)
g={m:A[A.model==m] for m in models}
for T in [0.95,0.98]:
    bs=[]
    for _ in range(1000):
        pick=rng.choice(models,len(models));sub=pd.concat([g[m] for m in pick]);bs.append(stat(sub,T))
    bs=np.array(bs);print('T',T,'cluster-boot 95% prec,rec,auc',[tuple(np.round(np.nanpercentile(bs[:,i],[2.5,97.5]),3)) for i in range(3)])
# by dataset
for ds,s in A.groupby(A.ds.str.split('/').str[-1].str.split('_').str[0]):
    for T in [0.95,0.98]:
        tp,fp,fn=conf(s,T);print(ds,len(s),T,'base %.2f prec %.3f rec %.3f auc %.3f'%((s.act>=T).mean(),tp/(tp+fp),tp/(tp+fn),auc(s.pred,s.act>=T)))
# leave out each model: range
for T in [0.95,0.98]:
    pr=[];rc=[]
    for m in models:
        s=A[A.model!=m];tp,fp,fn=conf(s,T);pr.append(tp/(tp+fp));rc.append(tp/(tp+fn))
    print('LOO model T',T,'prec',min(pr),max(pr),'rec',min(rc),max(rc))
# precision vs trivial baseline: classifier "all stages' single each >= T" i.e. predmin>=T, and "min"
for T in [0.95,0.98]:
    for nm in ['predmin','first']:
        sel=A[nm]>=T;pos=A.act>=T;print(T,nm,'prec',(pos&sel).sum()/sel.sum(),'rec',(pos&sel).sum()/pos.sum())
# per-model precision spread
for T in [0.98]:
    r=[]
    for m in models:
        s=g[m];tp,fp,fn=conf(s,T);r.append((m,len(s),tp,fp,fn))
    r=pd.DataFrame(r,columns=['m','n','tp','fp','fn']);r['prec']=r.tp/(r.tp+r.fp);print(r.sort_values('fp',ascending=False).head(8).to_string());print('models w fp>0',(r.fp>0).sum(),'of',len(r))
# conditional: restrict to hard cases where pred in [T-0.03,T+0.03]
for T in [0.95,0.98]:
    s=A[(A.pred>=T-0.03)&(A.pred<T+0.03)];tp,fp,fn=conf(s,T);print('near-threshold band T',T,len(s),'prec',tp/(tp+fp),'rec',tp/(tp+fn),'auc',auc(s.pred,s.act>=T))
# regret
print('--regret')
A['unit']=A.model+'|'+A.ds
for c0 in [10,20,40]:
    reg=[];w=[];n=0;rows=[]
    for u,s in A[A.comp>=c0].groupby('unit'):
        pass
    # note: "pipelines" candidate set = combined pipelines only
    for u,s in A.groupby('unit'):
        s=s[s.comp>=c0]
        if len(s)<3:continue
        ch=s.loc[s.pred.idxmax()];reg.append(s.act.max()-ch.act);rows.append((u,reg[-1],len(s)))
    reg=np.array(reg);print(c0,len(reg),'mean %.4f max %.4f within.005 %.2f'%(reg.mean(),reg.max(),(reg<=0.005).mean()),'boot mean CI',np.round(np.percentile([rng.choice(reg,len(reg)).mean() for _ in range(2000)],[2.5,97.5]),4))
    print(sorted(rows,key=lambda x:-x[1])[:3])
# regret with ties/noise: oracle best also selected on noisy act -> optimistic; compare to random pick and to "best actual by split"
    rr=[]
    for u,s in A.groupby('unit'):
        s=s[s.comp>=c0]
        if len(s)<3:continue
        rr.append(s.act.max()-s.act.mean())
    print('  random-pick regret mean',np.mean(rr))
    # smallest-candidate-set effect
    nn=[len(s[s.comp>=c0]) for u,s in A.groupby('unit') if len(s[s.comp>=c0])>=3];print('  cand set size median',np.median(nn),'min',min(nn))
# regret restricted: ties in act where within CI
# within-unit spearman
r=[]
for u,s in A.groupby('unit'):
    if len(s)>=10 and s.pred.nunique()>2 and s.act.nunique()>2:r.append(spearmanr(s.pred,s.act)[0])
r=np.array(r);print('spearman n',len(r),'median',np.median(r),'min',r.min())
