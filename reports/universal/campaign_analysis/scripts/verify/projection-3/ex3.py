from load import *
print(len(U),[k for k,v in U.items() if len(v)>1])
for (m,ds),v in sorted(U.items()):
    d=v[0][1]; R_=rows(d)
    if d['baseline']['dim']==128 and ds=='docvqa' and m.startswith(('5CD','Cog')):
        for l in R_:
            if 'project' in l and l.count('>')<2: print(m,l,R_[l][M],R_[l]['vectors_per_doc'])
