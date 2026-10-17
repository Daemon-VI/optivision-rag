from load import *
for (m,ds),v in U.items():
    if '4b' in m.lower() and 'nemotron' in m.lower():
        d=v[0][1]; print(m,ds,v[0][0][-80:],d['baseline']['dim'],[ (l,round(r[M],3)) for l,r in rows(d).items() if r['group']=='dim'])
