import sys; sys.path.insert(0,'.')
from load import *
pd.set_option('display.width',250); pd.set_option('display.max_columns',50)
F,C,O,K=load()
print(F.groupby('model').dataset.apply(list).to_string())
# candidate labels same across models?
labs=K.groupby('model').label.apply(lambda s: tuple(sorted(s)))
print('distinct candidate sets',labs.nunique())
print(K[K.model==K.model.iloc[0]][['label','family','compression','ret_all']].to_string())
O['sel_ok']=O.sel.notna()
O['met']=np.where(O.sel_ok, (O.hold_labels>=O.target).astype(float), np.nan)
O.to_pickle(OUT+'/O.pkl'); C.to_pickle(OUT+'/C.pkl'); F.to_pickle(OUT+'/F.pkl'); K.to_pickle(OUT+'/K.pkl')
# check summary vs outcomes
chk=O.groupby(['model','dataset','reference','rule','target']).agg(sel=('sel_ok','sum'),met=('met','mean')).reset_index()
m=C.merge(chk,on=['model','dataset','reference','rule','target'])
print('selected mismatch',(m.selected!=m.sel).sum(),'met_labels mismatch',(np.abs(m.met_labels.fillna(-1)-m.met_y.fillna(-1))>1e-9).sum() if 'met_y' in m else '')
print('met (holdout) vs met_labels differ in labels ref:',((C.reference=='labels')&(C.met.fillna(-1)!=C.met_labels.fillna(-1))).sum())
