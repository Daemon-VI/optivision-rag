exec(open(__file__.replace('v2.py','v.py')).read().split("for nm,only in")[0])
sets={'noproject':lambda l:'project' not in l,
'project128ok':lambda l:not re.search(r'project\((16|32|64)',l) and 'project(32, random)' not in l,
'only project':lambda l:'project' in l,
'proj16/32/64 only':lambda l:re.search(r'project\((16|32|64)',l) is not None,
'merge incl combos':lambda l:'merge' in l and 'project' not in l}
for nm,o in sets.items():
    v=[]
    out=[]
    for a,b in pairs[:8]:
        m,r,n=pair(a,b,DS,o); v.append(m); out.append(f"{m:+.2f}({n})")
    v=np.array(v); bs=[rng.choice(v,8).mean() for _ in range(5000)]
    print(nm,out,'mean %.2f CI %.2f..%.2f worse %d/8'%(v.mean(),*np.percentile(bs,[2.5,97.5]),(v>0).sum()))
