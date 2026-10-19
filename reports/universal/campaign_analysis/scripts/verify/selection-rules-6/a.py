import json,glob,os,numpy as np
R="C:/Users/Lenovo/projects/optivision-package/reports/universal/campaign/"
cells=0;abc=0;mn=9;ablist=[];lowc=0
for f in glob.glob(R+'*/selection_rules/*.json'):
    for x in json.load(open(f))['results']:
        if x['reference']=='baseline@1' and x['rule']=='calibrate() default':
            cells+=1
            ab=sum(o['selected'] is None for o in x['outcomes'])
            if ab: abc+=1; ablist.append((os.path.basename(f)[:30],x['target'],ab))
            for o in x['outcomes']:
                if o['selected'] is not None: mn=min(mn,o['holdout_labels']-x['target'])
print(cells,abc,mn); print(ablist)
