from load import *
import glob,os
fs=glob.glob('C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/*/wide/*.json')
print(len(fs)); print(fs[0])
j=json.load(open(fs[0])); print(j.keys()); print(j['baseline']); print(j['rows'][0].keys())
print([r['label'] for r in j['rows'] if 'merge' in r['label'] or 'ward' in r['label']][:40])
df,d=load()
# older models: in measured but not campaign?
camp={os.path.basename(os.path.dirname(os.path.dirname(f))) for f in fs}
print(len(camp))
print(sorted(set(m.replace('/','--') for m in df.model)-camp))
