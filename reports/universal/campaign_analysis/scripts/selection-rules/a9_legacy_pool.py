import sys; sys.path.insert(0,'.')
from load import *
import glob,re,pathlib
pd.set_option('display.width',250)
U='C:/Users/Lenovo/projects/optivision-package/reports/universal'
rm=open(U+'/campaign/README.md',encoding='utf8').read()
yes=[m.group(1) for m in re.finditer(r'^\| `([^`]+)` \|.*\| yes \|$',rm,re.M)]
F=pd.read_pickle(OUT+'/F.pkl'); have=set(F.model)
print('README yes',len(yes),'loaded',len(have),'yes but no file',[m for m in yes if m not in have],'file but not yes',[m for m in have if m not in yes])
orph=U+'/campaign/batches/11_text/orphans/colbertv2.0/selection_rules/colbertv2.0_scifact.json'
d=json.load(open(orph)); print('orphan',d['dataset'],d['n_queries'])
for f in glob.glob(U+'/campaign/*colbert*v2.0/selection_rules/*.json'): print(f, json.load(open(f))['n_queries'])
# legacy pool
rows=[]
for f in sorted(glob.glob(U+'/selection_rules/*.json')):
    n=pathlib.Path(f).stem
    if n.startswith('E2-colpali'): continue
    d=json.load(open(f))
    for r in d['results']:
        if r['reference']!='labels': continue
        for o in r['outcomes']:
            rows.append(dict(f=n,rule=r['rule'],target=r['target'],sel=o['selected'],comp=o.get('compression'),hold=o.get('holdout')))
L=pd.DataFrame(rows); L['ok']=L.sel.notna(); L['fail']=np.where(L.ok,(L.hold<L.target).astype(float),np.nan); L['logc']=np.log(L.comp)
g=L.groupby('rule').agg(n=('ok','size'),abst=('ok',lambda s:1-s.mean()),fail=('fail','mean'),gm=('logc',lambda s:np.exp(s.mean())))
print('legacy files',L.f.nunique(),'(excl E2-colpali)'); print(g.round(4))
