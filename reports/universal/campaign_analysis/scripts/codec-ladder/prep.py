import json,re
from load import *
reg=json.load(open('registry.json'))
def table(metric='ret'):
    df=load()
    df=df[df.codec!='float32']
    return df
def wide(metric='ret'):
    df=table()
    w=df.pivot_table(index=['model','dataset'],columns='codec',values=metric)[CODECS]
    meta=df.drop_duplicates(['model','dataset']).set_index(['model','dataset'])[['dim','vpd','base','dtype','n_docs','n_q','src']]
    w=w.join(meta).reset_index()
    w['ds']=w.dataset.str.extract(r'_(docvqa|infovqa|scifact)')[0]
    w['family']=w.model.map(lambda m: reg[m]['family'] if m in reg else None)
    w['fp16']=w.dtype.isin(['float16','torch.float16'])
    w['modality']=np.where(w.ds=='scifact','text','image')
    return w
if __name__=="__main__":
    w=wide(); print(w.shape); print(w.fp16.sum(), w.groupby('ds').size()); print(w[w.dtype=='None'].model.unique()); print(w.family.isna().sum())
    pd.set_option('display.width',250); print(w.groupby('fp16')[CODECS].mean())
