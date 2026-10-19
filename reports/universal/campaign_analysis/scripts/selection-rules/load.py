import json,glob,os,re,pathlib
import numpy as np,pandas as pd
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign'
OUT='C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/selection-rules'
def load():
    cells=[];outs=[];files=[];cands=[]
    for f in sorted(glob.glob(R+'/*/selection_rules/*.json')):
        f=pathlib.Path(f).as_posix(); model=f.split('/')[-3].replace('--','/',1)
        d=json.load(open(f))
        ds='scifact' if 'scifact' in d['dataset'] else ('docvqa' if 'docvqa' in d['dataset'] else 'infovqa')
        wide=glob.glob(f"{R}/{f.split('/')[-3]}/wide/*_{'scifact' if ds=='scifact' else ds+'_test_subsampled'}.json")
        base=None;vpd=None;dim=None
        if wide:
            w=json.load(open(wide[0])); b=w['baseline']; base=b.get('ndcg@5'); vpd=b.get('vectors_per_doc'); dim=b.get('dim')
            bm=b.get('ndcg@10')
        files.append(dict(model=model,dataset=ds,file=f,n_docs=d['n_docs'],n_queries=d['n_queries'],metric=d['metric'],n_cand=d['n_candidates'],baseline_ndcg5=base,baseline_ndcg10=bm if wide else None,vpd=vpd,dim=dim,seconds=d.get('seconds')))
        cmap={c['label']:c for c in d['candidates']}
        for c in d['candidates']: cands.append(dict(model=model,dataset=ds,label=c['label'],family=c['family'],compression=c['compression'],ret_all=c['retention_labels_all']))
        for r in d['results']:
            cells.append(dict(model=model,dataset=ds,reference=r['reference'],rule=r['rule'],target=r['target'],selected=r['selected'],splits=r['splits'],met=r['met'],met_labels=r['met_labels'],holdout_mean=r['holdout_mean'],holdout_min=r['holdout_min'],comp_med=r['compression_median']))
            for o in r['outcomes']:
                x=dict(model=model,dataset=ds,reference=r['reference'],rule=r['rule'],target=r['target'],seed=o['seed'],sel=o['selected'],judged=o['judged'])
                if o['selected']:
                    x.update(comp=o['compression'],cal=o['calibration'],hold=o['holdout'],hold_labels=o['holdout_labels'],ret_all=cmap[o['selected']]['retention_labels_all'] if o['selected'] in cmap else np.nan, fam=cmap[o['selected']]['family'] if o['selected'] in cmap else None)
                outs.append(x)
    return pd.DataFrame(files),pd.DataFrame(cells),pd.DataFrame(outs),pd.DataFrame(cands)
if __name__=='__main__':
    F,C,O,K=load()
    print(F.describe(include='all').T.to_string()); print(len(F),len(C),len(O),len(K))
    print(F.groupby('dataset').size()); print(F.n_cand.value_counts()); print(F.model.nunique())
    print(F[F.baseline_ndcg5.isna()])
    F.to_csv(OUT+'/files.csv',index=False)
