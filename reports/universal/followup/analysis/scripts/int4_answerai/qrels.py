import json,numpy as np
for n,f in [('nfcorpus','f1_text_nfcorpus_a/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_nfcorpus_qrels.json'),('scifact','f3b_answerai_scifact/answerdotai--answerai-colbert-small-v1/answerai-colbert-small-v1_scifact_qrels.json')]:
    q=json.load(open(f)); c=np.array([len(v) for v in q.values()])
    print(n,len(q),'rel/query mean',c.mean().round(1),'median',np.median(c),'frac==1',(c==1).mean().round(2),'max',c.max())
