import sys;sys.path.insert(0,r"C:/Users/Lenovo/AppData/Local/Temp/claude/C--Users-Lenovo/07215cc9-ee09-45ef-91e1-819ae4093530/scratchpad/int4norm_analysis/reproducibility")
from load import *
import numpy as np
R2=R+'followup/'
# per-unit seconds
rowsout=[]
for (b,m,ds),f in sorted(s17().items()):
    d=json.load(open(f))
    # matching s13 seconds from f9_..f13 first-run batches (same prefix without b)
    b0=b.replace('b_','_',1); 
    g=R2+f"{b0}/{m}/wide/{m.split('--')[1]}_{ds}.json"
    s13=None; n13=None
    if os.path.exists(g):
        e=json.load(open(g)); s13=e['seconds']; n13=len(e['rows'])
    rowsout.append((b[:4],m.split('--')[1],ds,d['n_docs'],d['n_queries'],len(d['rows']),d['seconds'],s13,n13))
print('batch model dataset n_docs nq rows s17sec sec/row s13sec s13rows')
for r in rowsout:
    print(r[0],f"{r[1][:28]:28}",f"{r[2][:12]:12}",r[3],r[4],r[5],f"{r[6]:.0f}",f"{r[6]/r[5]:.1f}",'|', (f"{r[7]:.0f} {r[8]}" if r[7] else '-'))
x=np.array([r[3] for r in rowsout],float);y=np.array([r[6]/r[5] for r in rowsout])
print('corr(n_docs, s/row) %.2f ; corr log-log %.2f'%(np.corrcoef(x,y)[0,1],np.corrcoef(np.log(x),np.log(y))[0,1]))
sl=np.polyfit(np.log(x),np.log(y),1);print('loglog slope %.2f'%sl[0])
# paired where both
pr=[(r[6],r[7]) for r in rowsout if r[7]]
print('paired n=%d  s17 total %.0f  s13 total %.0f ; median ratio s17/s13 %.2f'%(len(pr),sum(a for a,b in pr),sum(b for a,b in pr),np.median([a/b for a,b in pr])))
print('s17 rows 21 vs s13 rows',sorted({r[8] for r in rowsout if r[8]}))
# run.json steps
print('\nrun.json steps')
for rj in sorted(glob.glob(R2+'f*b_*/*/run.json')):
    rj=rj.replace(chr(92),'/');j=json.load(open(rj));st=j['steps']
    enc=sum(v['seconds'] for k,v in st.items() if k.startswith('encode'));s17s=sum(v['seconds'] for k,v in st.items() if k.startswith('s17'));ge=sum(v['seconds'] for k,v in st.items() if k.startswith('geometry'))
    other={k:v['status'] for k,v in st.items() if v['status']!='ok'}
    print(rj.split('/')[-3][:5],rj.split('/')[-2][:34].ljust(34),f"enc {enc:.0f} geom {ge:.0f} s17 {s17s:.0f}",other or '')
