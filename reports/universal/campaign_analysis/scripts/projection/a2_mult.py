"""A2: do merge>project and pure project agree? (independence of the two effects)"""
from common import *
c=curve.dropna(subset=['rel_pure','rel_merge']).copy()
c['diff']=c.rel_merge-c.rel_pure
print('n pairs',len(c),'models',c.model.nunique())
print(c.groupby(['D','k'])['diff'].agg(['mean','std','count']).round(4))
e,lo,hi=cluster_boot(c['diff'],c.model,B=5000)
print('overall mean diff (merge-controlled minus pure) %.4f CI [%.4f, %.4f]'%(e,lo,hi), 'mean abs %.4f'%c['diff'].abs().mean(), 'max abs %.4f'%c['diff'].abs().max())
print(np.corrcoef(c.rel_merge,c.rel_pure)[0,1])
