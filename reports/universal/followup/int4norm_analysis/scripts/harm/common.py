import json,glob,os
import numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup/'
BATCHES=['f9b_int4norm_text','f10b_int4norm_pages_128d','f11b_int4norm_pages_wide','f12b_text_scidocs','f13b_text_scale_same_domain']
def load():
    U={}
    for b in BATCHES:
        for p in sorted(glob.glob(R+b+'/*/int4norm/*.json')):
            d=json.load(open(p)); model=p.replace(chr(92),'/').split('/')[-3]
            U[(model.split('--')[1],d['dataset'].split(model.split('--')[1]+'_')[1])]=dict(d=d,batch=b,path=p)
    return U
def row(d,l): return next(r for r in d['rows'] if r['label']==l)
