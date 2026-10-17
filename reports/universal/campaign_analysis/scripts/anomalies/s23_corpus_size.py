from common import *
import pandas as pd
from scipy import stats
P=PK+'/reports/universal/audit/'
for fn in ('corpus_size.json','colqwen_corpus_size.json'):
    d=pd.DataFrame(json.load(open(P+fn)))
    print(fn,len(d),'rows; sizes',sorted(d.corpus_size.unique()),'draws',d.draw.nunique(), 'datasets',d.dataset.unique(), 'queries',sorted(d.queries.unique()))
    g=d.groupby(['dataset','config','corpus_size']).retention.mean().unstack()
    print(g.round(3).to_string())
    # slope of retention on log size, per config, with draw-clustered view
    for (ds,cf),s in d.groupby(['dataset','config']):
        x=np.log(s.corpus_size); r=stats.linregress(x,s.retention)
        print(f'  {ds:8s}{cf:26s} slope per doubling={r.slope*np.log(2):+.4f} (se {r.stderr*np.log(2):.4f}) n={len(s)}')
