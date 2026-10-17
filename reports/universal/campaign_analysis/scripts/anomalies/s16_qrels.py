from common import *
import collections, glob, hashlib
fs=glob.glob(CAMP+'/*/*_qrels.json')
print(len(fs),'qrels files')
h=collections.defaultdict(set); ntarget={}
for f in fs:
    d=json.load(open(f)); ds='docvqa' if 'docvqa' in f else 'infovqa'
    h[ds].add(hashlib.md5(json.dumps(d,sort_keys=True).encode()).hexdigest())
    ntarget[ds]=(len(d),len({x for v in d.values() for x in v}),collections.Counter(len(v) for v in d.values()))
print({k:len(v) for k,v in h.items()}, ntarget)
d=json.load(open(fs[0]))
for ds in ('docvqa','infovqa'):
    f=[x for x in fs if ds in x][0]; d=json.load(open(f))
    c=collections.Counter(x for v in d.values() for x in v)
    print(ds,'queries',len(d),'distinct pages',len(c),'pages with >1 query',sum(1 for v in c.values() if v>1),'max',max(c.values()))
    # docs per source doc (page 'id::pN')
    src=collections.Counter(x.split('::')[0] for x in c); print(' source docs',len(src),'max pages from one doc',max(src.values()))
