import json,numpy as np
from scipy import stats
M=json.load(open('C:/Users/Lenovo/projects/optivision-package/src/optivision/presets/measured.json'))['models']
dv={};iv={}
for m,d in M.items():
    for ds,v in d.items():
        if 'docvqa' in ds.lower(): dv[m]=v['baseline']
        if 'infovqa' in ds.lower(): iv[m]=v['baseline']
c=sorted(set(dv)&set(iv));print(len(c))
x=np.array([dv[m] for m in c]);y=np.array([iv[m] for m in c]);print(stats.spearmanr(x,y)[0])
lg=lambda p:np.log(p/(1-p));X=lg(y);Y=lg(x);zs=[]
for i in range(len(c)):
    k=np.arange(len(c))!=i;b=np.polyfit(X[k],Y[k],1);r=Y[k]-np.polyval(b,X[k])
    zs.append((Y[i]-np.polyval(b,X[i]))/r.std(ddof=2))
zs=np.array(zs);j=np.argsort(-abs(zs))[:4];print([(c[i],round(zs[i],2)) for i in j])
# bootstrap codec int8 clustered handled elsewhere; sign test
from scipy.stats import binomtest
print(binomtest(21,47).pvalue)
