import re,numpy as np
L=[l for l in open('a5.out') if l.startswith(('f9b_','f12b','f13b'))]
D={}
for l in L:
    p=l.split('|')[0].split()
    D[(p[1],p[2][:8])]=(int(p[3]),float(p[7]))
print('text-only units',len(D))
x=np.log([v[0] for v in D.values()]);y=np.log([v[1] for v in D.values()]);print('text loglog slope %.2f'%np.polyfit(x,y,1)[0])
for m in ['answerai-colbert-small-v1','colbertv2.0']:
    print(m,{k[1]:v for k,v in D.items() if k[0]==m})
    d=[(v[0],v[1]) for k,v in D.items() if k[0]==m]; d.sort(); print(' slope %.2f'%np.polyfit(np.log([a for a,b in d]),np.log([b for a,b in d]),1)[0])
