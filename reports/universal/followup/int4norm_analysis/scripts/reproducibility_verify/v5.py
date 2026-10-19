import json,glob,re,numpy as np
R='C:/Users/Lenovo/projects/optivision-package/reports/universal'
def sc(b):
    o={}
    for p in glob.glob(b+'/**/*.json',recursive=True):
        p=p.replace(chr(92),'/')
        if 'run.json' in p or '/geometry/' in p: continue
        try:d=json.load(open(p))
        except:continue
        if isinstance(d,dict) and 'rows' in d and 'seconds' in d:
            m=p.split(b+'/')[1].split('/')[0]; o[(m,d['dataset'],d['n_docs'])]=(d['seconds'],len(d['rows']))
    return o
n={};
for b in ['f9b_int4norm_text','f10b_int4norm_pages_128d','f11b_int4norm_pages_wide','f12b_text_scidocs','f13b_text_scale_same_domain']: n.update(sc(f'{R}/followup/{b}'))
o={}
for b in glob.glob(R+'/followup/f*_*')+[R+'/campaign']:
    b=b.replace(chr(92),'/')
    if re.search(r'f\d+b_',b) and 'f3b' not in b: continue
    for k,v in sc(b).items(): o.setdefault(k,[]).append(v)
pairs=[(k,n[k],o[k]) for k in n if k in o]
print(len(n),len(pairs))
s17=sum(p[1][0] for p in pairs); 
s13=sum(max(x[0] for x in p[2]) for p in pairs); s13f=sum(p[2][0][0] for p in pairs)
print(s17,s13,s13f,np.median([p[1][0]/max(x[0] for x in p[2]) for p in pairs]))
print(set(p[1][1] for p in pairs), sorted(set(x[1] for p in pairs for x in p[2]))[:3],max(x[1] for p in pairs for x in p[2]))
for k in n:
    if 'answerai' in k[0] and k[1]=='scidocs' or 'LateOn' in k[0] and k[1].endswith('scidocs') and 'm' not in k[0][9:10]: print(k,n[k],o.get(k))
print('--')
for k in n:
    if k[1].endswith('scidocs') and ('answerai' in k[0] or k[0].endswith('LateOn')): print(k,n[k],o.get(k))
