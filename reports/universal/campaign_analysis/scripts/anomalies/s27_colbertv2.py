from common import *
import pandas as pd
from scipy import stats
M=measured()
a=M['colbert-ir/colbertv2.0']['BeIR/scifact']; b=M['lightonai/colbertv2.0']['BeIR/scifact']
ra={r['label']:r for r in a['rows']}; rb={r['label']:r for r in b['rows']}
L=[l for l in ra if l in rb and l!='float32']
d=np.array([ra[l]['retention']-rb[l]['retention'] for l in L])
print('baseline ndcg@10',a['baseline'],b['baseline'],'diff',a['baseline']-b['baseline'],' vpd',ra['float32']['vectors_per_doc'],rb['float32']['vectors_per_doc'],'bytes/doc',ra['float32']['bytes_per_doc'],rb['float32']['bytes_per_doc'])
print('labels',len(L),'mean(colbert-ir - lightonai) retention',d.mean().round(4),'sd',d.std(ddof=1).round(4),'sign up/down',(d>0).sum(),(d<0).sum(),'wilcoxon p',stats.wilcoxon(d).pvalue.round(4))
# absolute bytes comparison: same label, bytes
for l in ['int4(mean)','binary','hierarchical_merge(0.25) > int4(mean)']:
    print(l,'bytes',round(ra[l]['bytes_per_doc']),round(rb[l]['bytes_per_doc']),'comp',round(ra[l]['compression'],2),round(rb[l]['compression'],2),'ret',round(ra[l]['retention'],3),round(rb[l]['retention'],3))
# absolute ndcg@10 of pipeline: baseline*retention
ab=[a['baseline']*ra[l]['retention']-b['baseline']*rb[l]['retention'] for l in L]; print('absolute ndcg@10 diff mean',np.mean(ab).round(4))
import pandas as pd
H=pd.DataFrame(json.load(open("C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/hubrank.json")))
t=H.downloads.sum(); print('total listed 30d downloads',t,'colbert-ir share',round(H.set_index('id').downloads['colbert-ir/colbertv2.0']/t,3),'lightonai/colbertv2.0',H.set_index('id').downloads.get('lightonai/colbertv2.0'))
# rank by downloads of those within catalogue 
print(H[H.id.isin(M)].sort_values('downloads',ascending=False).head(5)[['id','downloads']].to_string(index=False))
# text retention metric choice effect
W=wide_files(); rows=[]
for (fo,fn),w in W.items():
    if 'scifact' not in fn: continue
    for r in w['rows']:
        rows.append((fo,r['label'],r['retention:ndcg@5'],r['retention:ndcg@10'],r['compression_vs_float32']))
t=pd.DataFrame(rows,columns=['m','l','r5','r10','c']); t['d']=t.r10-t.r5
print('text rows',len(t),'mean r10-r5',t.d.mean().round(4),'sd',t.d.std().round(4),'min/max',t.d.min().round(3),t.d.max().round(3))
for thr in (0.95,0.99):
    flip=((t.r5>=thr)!=(t.r10>=thr)); print('threshold',thr,'rows whose pass/fail flips',flip.sum(),'of',len(t),' of rows passing under either',((t.r5>=thr)|(t.r10>=thr)).sum())
