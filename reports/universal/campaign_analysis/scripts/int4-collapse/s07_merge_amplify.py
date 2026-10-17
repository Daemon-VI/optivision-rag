import pandas as pd,numpy as np
from scipy import stats
t=pd.read_pickle('merge_int4.pkl'); df=pd.read_pickle('master2.pkl')
t['relc']=t.m_int4c/t['merge']; t['mod']=np.where(t.dataset=='_scifact','text','image')
t['d_plain']=t.rel-t.plain
NT=0
for mod,g in t.groupby('mod'):
    w=stats.wilcoxon(g.rel,g.plain); NT+=1
    print(mod,'rows',len(g),'median rel',g.rel.median().round(3),'median plain',g.plain.median().round(3),'median d',g.d_plain.median().round(4),'frac rel<plain-0.01:',(g.d_plain<-0.01).mean().round(2),'Wilcoxon p',w.pvalue)
print('centred relc: min',t.relc.min().round(3),'rows<0.97',(t.relc<0.97).sum(),'of',len(t)); print(t[t.relc<0.97][['model','q','merge','m_int4c','relc']].round(3).to_string())
# Per-model fragility under merging for text (single dataset): rel at 0.25
tx=t[(t['mod']=='text')&(t.q=='0.25')].sort_values('rel')
print(tx[['model','plain','merge','m_int4','rel','relc','m_bin','bin']].round(3).to_string())
# does merging 0.25 (more pooled) hurt more than 0.33: paired sign
p=t.pivot_table(index=['model','dataset'],columns='q',values='rel'); 
for sub,idx in [('text',p.index.get_level_values(1)=='_scifact'),('image',p.index.get_level_values(1)!='_scifact')]:
    q=p[idx]; dd=(q['0.25']-q['0.33']); 
    print(sub,'rel(0.25)<rel(0.33):',(dd<-0.002).sum(),'>',(dd>0.002).sum(),'n',len(dd),'median',dd.median().round(4)); NT+=1
# plain-int4 deficits that do exist among image models after merging
im=t[t['mod']=='image']; print(im.sort_values('rel').head(6)[['model','dataset','q','plain','merge','m_int4','rel']].round(3).to_string())
# Spearman vulnerability (rel at 0.25 for text) vs plain
r=stats.spearmanr(tx.plain,tx.rel); print('text Spearman plain~rel(0.25)',r); NT+=1
print('tests',NT)
