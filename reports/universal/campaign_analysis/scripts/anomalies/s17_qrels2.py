from common import *
import collections, glob, hashlib
fs=[f for f in glob.glob(CAMP+'/*/*infovqa*_qrels.json')]
groups=collections.defaultdict(list)
for f in fs:
    d=json.load(open(f)); groups[hashlib.md5(json.dumps(d,sort_keys=True).encode()).hexdigest()].append((f.replace(os.sep,'/').split('/')[-2],d))
for k,v in groups.items(): print(k[:8],len(v),[x[0] for x in v][:60] if len(v)<5 else '')
ks=list(groups); a=groups[ks[0]][0][1]; b=groups[ks[1]][0][1]
diff=[q for q in a if a[q]!=b.get(q)]; print(len(diff),len(a),len(b)); print([(q,a[q],b.get(q)) for q in diff[:5]])
print(len(set(a)^set(b)))
# which dataset list the 'wide' json for both
