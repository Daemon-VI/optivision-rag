exec(open('a1.py').read().split("for scope in")[0])
rng=np.random.default_rng(0)
def boot(x,f=np.median):
    x=np.array(x);return np.percentile([f(rng.choice(x,len(x))) for _ in range(2000)],[2.5,97.5]).round(3)
# failing pipeline per target
for t in[.95,.97,.99]:
    ok=[l for l in cand if all(units[p][l]['retention']>=t for p in ev)]
    print(t,'passes by all 88 units:',[(l,round(np.median([units[p][l]['compression'] for p in ev]),1)) for l in ok if units[ev[0]][l]['compression']>1])
    # peers failing int4(mean) at .97
    for l in ['int4(mean)','int8(per_vector)']:
        print('  ',l,'fails',[(p[0].split('/')[1][:20],p[1][7:10],round(units[p][l]['retention'],3)) for p in ev if units[p][l]['retention']<t])
for t in[.95,.97,.99]:
  for name,kw in [('base',{}),('lowerCI',{'lower':True}),('noSib',{'exclude_sib':True}),('margin.01',{'margin':.01})]:
    o=run(t,'all',**kw)
    hit=[x[2] for x in o];fr=[x[3]/x[4] for x in o]
    print(t,name,round(np.mean(hit),3),'med comp',round(np.median([x[3] for x in o]),1),'frac(all units) med',round(np.median(fr),3),boot(fr),Counter(x[1] for x in o).most_common(2))
# oracle robustness: oracle with lower CI >= t on own unit (conservative oracle)
for t in[.95,.97,.99]:
    fr=[];
    for k in ev:
        u=units[k]; ob=max(u[l]['compression'] for l in cand if u[l]['retention_interval'][0]>=t)
        fr.append(u['int4(mean)']['compression']/ob if t<.97 else 0)
    print('oracle lowerCI',t,round(np.median([max(units[k][l]['compression'] for l in cand if units[k][l]['retention_interval'][0]>=t) for k in ev]),1))
# by dataset, and float16 baseline units
for t in[.95]:
    o=run(t,'all')
    for g in ['vidore/d','vidore/i','BeIR/sci']:
        s=[x for x in o if x[0][1].startswith(g)]
        print(g,len(s),round(np.median([x[3]/x[4] for x in s]),3),round(np.median([x[4] for x in s]),1))
# drop top-5 oracle models
