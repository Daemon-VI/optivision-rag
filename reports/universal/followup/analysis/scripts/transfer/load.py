import json,glob,os
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
def load():
    out={}
    for p in glob.glob(R+'followup/f[12]_*/*/wide/*nfcorpus.json'):
        m=os.path.basename(os.path.dirname(os.path.dirname(p)))
        sp=R+'campaign/'+m+'/wide/'
        c=glob.glob(sp+'*scifact.json')
        if m.startswith('answerdotai'): c=glob.glob(R+'followup/f3b*/'+m+'/wide/*scifact.json')
        if not c: print('missing',m); continue
        a=json.load(open(c[0])); b=json.load(open(p))
        out[m]=(a,b)
    return out
