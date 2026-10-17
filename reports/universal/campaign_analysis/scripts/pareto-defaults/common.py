from load import *
import warnings; warnings.filterwarnings('ignore')
from optivision.adapters import MODELS
ALIAS={'none > float32':'float32','none > float16':'float16','none > int8/vec':'int8(per_vector)','none > int4':'int4','none > binary':'binary'}
def get():
    m=measured(); m['key']=m.model+'|'+m.dataset
    m['label0']=m.label; m['label']=m.label.replace(ALIAS)
    m=m[~m.label0.str.startswith('ward')].copy()
    m['modal']=np.where(m.dataset.str.contains('scifact'),'text','image')
    m['ds']=m.dataset.str.extract(r'(docvqa|infovqa|scifact)')[0]
    m['family']=m.model.map(lambda x:MODELS[x].family)
    m['dim']=m.model.map(lambda x:MODELS[x].dimension)
    m['legacy']=m.key.isin(set(m[m.label0.str.startswith('none')].key))
    return m
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)
