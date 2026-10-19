from common import *
from scipy.stats import fisher_exact, spearmanr
rng=np.random.default_rng(5)
df=long_df()
P=mat(df)
I4=P['int4'].dropna()
# per model min over datasets
mod=I4.groupby(level='model').agg(['min','max','count'])
mod['collapse']=mod['max']<0.9
mod['group']=[GROUP2.get(m,'single') for m in mod.index]
mod['bb']=[GROUP.get(m,'text' if info(m)['modality']=='text' else 'single') for m in mod.index]
print('models with int4:',len(mod),'collapsed:',int(mod.collapse.sum()))
print(mod[mod.collapse | (mod['min']<0.95)].round(3))
print('consistency across datasets for image models: max-min of int4 retention for collapsed',(mod[mod.collapse]['max']-mod[mod.collapse]['min']).round(3).to_dict())
def sibtest(groupcol):
    # for each model with >=1 sibling: indicator sibling_collapse (any other member collapsed) vs own collapse
    rows=[]
    for m,r in mod.iterrows():
        g=r[groupcol]
        if g=='single': continue
        sib=mod[(mod[groupcol]==g)&(mod.index!=m)]
        if len(sib)==0: continue
        rows.append((m,bool(r.collapse),bool(sib.collapse.any()),len(sib)))
    t=pd.DataFrame(rows,columns=['m','c','sc','n'])
    tab=pd.crosstab(t.c,t.sc).reindex(index=[False,True],columns=[False,True],fill_value=0)
    o,p=fisher_exact(tab.values)
    return tab,o,p,len(t)
for col in ['group','bb']:
    tab,o,p,n=sibtest(col)
    print(col,'n',n,'\n',tab,'\n Fisher OR',o,'p',p)
# severity: text models, relation of int4 vs binary / lloyd2 / dim
T=P.xs('sci',level='ds')[['int4','binary','lloyd2(7)','int4(mean)','hierarchical_merge(0.1)']].dropna()
T['dim']=[info(m)['dim'] for m in T.index]
print(T.round(3))
for c in ['binary','lloyd2(7)','dim','hierarchical_merge(0.1)']:
    r,p=spearmanr(T['int4'],T[c]); print('text spearman int4 vs',c,round(r,3),round(p,4),'n',len(T))
# image models: any relation of int4 retention to dim?
Im=P.drop(index='sci',level='ds')['int4'].dropna()
print('image int4 min',Im.min(),'n collapsed',(Im<0.9).sum(),'of',len(Im))
print('image int4(mean) range',P.drop(index='sci',level='ds')['int4(mean)'].min(),P.drop(index='sci',level='ds')['int4(mean)'].max())
print('text int4(mean) range',P.xs('sci',level='ds')['int4(mean)'].min(),P.xs('sci',level='ds')['int4(mean)'].max())
# within backbone qwen3vl
q=mod[mod.bb=='qwen3vl']; print(q.round(3))
# nemotron dim vs EVIE: dims
for m in ['nvidia/nemotron-colembed-vl-4b-v2','nvidia/nemotron-colembed-vl-8b-v2','tencent/EVIE-4.5B','tencent/EVIE-8B']: print(m,info(m)['dim'])
