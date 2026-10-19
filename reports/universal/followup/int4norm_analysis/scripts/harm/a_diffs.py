import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
U=load(); print(len(U),'units')
out=[]
for k,u in U.items():
    d=u['d']
    for a,b in [('int4norm','int4'),('int4norm(mean)','int4(mean)')]:
        for m in ['ndcg@5','ndcg@10']:
            ra,rb=row(d,a),row(d,b)
            out.append((k,a+' - '+b,m,ra['retention:'+m]-rb['retention:'+m],ra['retention:'+m],rb['retention:'+m],u['batch']))
print('negative diffs (retention units), any size'); 
neg=[o for o in out if o[3]<0]
print(len(neg),'of',len(out),'cells negative')
for o in sorted(neg,key=lambda o:o[3])[:40]: print('%-45s %-28s %-8s %+.4f (%.4f vs %.4f) %s'%(o[0],o[1],o[2],o[3],o[4],o[5],o[6][:4]))
print('\nneg <= -0.005:')
for o in sorted([o for o in neg if o[3]<=-0.005],key=lambda o:o[3]): print('%-45s %-28s %-8s %+.4f'%(o[0],o[1],o[2],o[3]))
import collections
for pair in ['int4norm - int4','int4norm(mean) - int4(mean)']:
  for m in ['ndcg@5','ndcg@10']:
    v=np.array([o[3] for o in out if o[1]==pair and o[2]==m]); 
    print(pair,m,'n=%d neg=%d median=%+.4f mean=%+.4f min=%+.4f'%(len(v),(v<0).sum(),np.median(v),v.mean(),v.min()))
# excluding units where int4 is badly broken (retention<0.9)
for pair,base in [('int4norm - int4','int4'),('int4norm(mean) - int4(mean)','int4(mean)')]:
    sel=[o for o in out if o[1]==pair and o[2]=='ndcg@5' and o[5]>=0.95]
    v=np.array([o[3] for o in sel]); print(pair,'units where baseline codec>=0.95:',len(v),'neg',(v<0).sum(),'mean %+.4f sd %.4f'%(v.mean(),v.std(ddof=1)))
