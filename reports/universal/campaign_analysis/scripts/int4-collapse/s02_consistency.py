"""Data issue check: wide-file retention vs measured.json retention for int4 etc."""
import json,pandas as pd,numpy as np
df=pd.read_pickle('master2.pkl'); w=df[df.src=='wide']
M=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
out=[]
for _,r in w.iterrows():
    for dn,v in M[r.model].items():
        if dn.split('/')[-1].startswith(r.dataset):
            for x in v['rows']:
                if x['label'] in ('int4','int4(mean)','binary','int8(per_vector)'):
                    key={'int4':'int4','int4(mean)':'int4(mean)','binary':'binary','int8(per_vector)':'int8(per_vector)'}[x['label']]
                    for met in ['ndcg@5','ndcg@10']:
                        out.append((r.model,r.dataset,x['label'],met,x['retention'],r[f'{key}|{met}'],v['baseline'],r.base5,r.base10))
o=pd.DataFrame(out,columns=['model','ds','label','met','measured','wide','mbase','b5','b10'])
for met in ['ndcg@5','ndcg@10']:
    s=o[o.met==met]; print(met,'max abs diff',(s.measured-s.wide).abs().max(),'n>0.002:',((s.measured-s.wide).abs()>0.002).sum(),'of',len(s))
b=o.drop_duplicates(['model','ds'])
print('baseline equals base5:',np.isclose(b.mbase,b.b5).sum(),'equals base10:',np.isclose(b.mbase,b.b10).sum(),'of',len(b))
bad=o[(o.met=='ndcg@5')&((o.measured-o.wide).abs()>0.002)]
print(bad.round(3).to_string())
b2=b[~np.isclose(b.mbase,b.b5)]; print(b2[['model','ds','mbase','b5','b10']].round(4).to_string())
print('---- by modality, max|measured - wide| per metric (rounded baselines tolerance 2e-4)')
o['modality']=np.where(o.ds=='scifact','text','image')
o['d']=(o.measured-o.wide).abs()
print(o.groupby(['modality','met']).d.agg(['max','mean']).round(4))
b['bm10']=np.isclose(b.mbase,b.b10,atol=2e-4); b['bm5']=np.isclose(b.mbase,b.b5,atol=2e-4)
print(b.assign(modality=np.where(b.ds=='scifact','text','image')).groupby('modality')[['bm5','bm10']].sum(), b.groupby(b.ds=='scifact').size())
