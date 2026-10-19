import json,glob,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/followup'
d=json.load(open(glob.glob(R+'/f9b_int4norm_text/answerdot*/run.json')[0])); print(json.dumps(d)[:1500])
def sec(p): return json.load(open(p))
for m,n in [('answerdotai--answerai-colbert-small-v1','answerai-colbert-small-v1'),('colbert-ir--colbertv2.0','colbertv2.0')]:
    pts=[]
    for b,ds in [('f9b_int4norm_text','nfcorpus'),('f9b_int4norm_text','scifact'),('f13b_text_scale_same_domain','scifact_plus10000'),('f12b_text_scidocs','scidocs')]:
        try: x=sec(f'{R}/{b}/{m}/int4norm/{n}_{ds}.json')
        except Exception as e: continue
        pts.append((x['n_docs'],x['seconds']/len(x['rows']),x['seconds'],len(x['rows'])))
    pts=np.array(pts);print(m,pts.tolist(), np.polyfit(np.log(pts[:,0]),np.log(pts[:,1]),1)[0])
