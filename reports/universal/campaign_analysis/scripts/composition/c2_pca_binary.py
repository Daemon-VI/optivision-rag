import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition")
from common import *
from scipy.stats import spearmanr
P=pd.read_pickle(r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/research/composition/pairs.pkl")
pd.set_option('display.width',250); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',300)
A=P[(P.pred_type=='all_singles')&(P.kinds=='project>codec')].copy(); A['err']=A.act-A.pred
A['codec']=A.stages.map(lambda s:s[1])
b=A[A.codec=='binary'].copy(); b['proj']=b.parts.map(lambda p:p[0]); b['bin']=b.parts.map(lambda p:p[1])
b['gain']=b.act-b.bin   # project>binary minus binary alone
b['grp']=np.where(b.ds=='scifact','text','image')
print(b[b.grp=='text'][['model','proj','bin','act','pred','err']].round(3).to_string())
print('text: share err>0',(b[b.grp=="text"].err>0).mean(),' image share err>0',(b[b.grp=="image"].err>0).mean())
# does projection help binary (act > binary alone) ?
for g,x in b.groupby('grp'):
    print(g,'n',len(x),'mean(act-binary_alone) %.4f'%x.gain.mean(),'share act>binary_alone %.2f'%(x.gain>0).mean(), 'spearman(err, binary alone loss)=%.2f p=%.2g'%spearmanr(x.err,1-x.bin))
# image: project k relative to dim
print(b[b.grp=='image'].assign(l=lambda d:1-d.bin).sort_values('l',ascending=False).head(8)[['model','proj','bin','act','pred','err']].round(3))
