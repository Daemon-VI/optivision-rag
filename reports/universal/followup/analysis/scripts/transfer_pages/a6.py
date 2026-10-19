import sys,os,json,glob;sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from load import *
R2=R+'followup/'
for m in ['f5_pages_wide/tencent--EVIE-4.5B','f4_pages_128d/vidore--colSmol-500M','f5_pages_wide/vultr--VultronRetrieverCore-Qwen3.5-4.5B']:
    for p in sorted(glob.glob(R2+m+'/geometry/*.json')):
        g=json.load(open(p)); print(os.path.basename(p)); 
        print({k:(v if not isinstance(v,dict) else '{..}') for k,v in g.items()} if len(str(g))<1500 else list(g.keys()))
        break
    break
g=json.load(open(glob.glob(R2+'f5_pages_wide/tencent--EVIE-4.5B/geometry/*tab*.json')[0]))
print(json.dumps(g,indent=0)[:3000])
