import json
R='C:/Users/Lenovo/projects/optivision-package/reports/universal/'
P=[('colbertv2','campaign/colbert-ir--colbertv2.0/wide/colbertv2.0_scifact.json','followup/f3_text_scale/colbert-ir--colbertv2.0/wide/colbertv2.0_scifact_plus10000.json'),
('answerai','followup/f3b_answerai_scifact/answerdotai--answerai-colbert-small-v1/wide/answerai-colbert-small-v1_scifact.json','followup/f3_text_scale/answerdotai--answerai-colbert-small-v1/wide/answerai-colbert-small-v1_scifact_plus10000.json'),
('colSmol','campaign/vidore--colSmol-500M/wide/colSmol-500M_docvqa_test_subsampled.json','followup/f6_scale_colsmol/vidore--colSmol-500M/wide/colSmol-500M_docvqa_test_subsampled_plus2000.json'),
('colpali','campaign/vidore--colpali-v1.3/wide/colpali-v1.3_docvqa_test_subsampled.json','followup/f7_scale_colpali/vidore--colpali-v1.3/wide/colpali-v1.3_docvqa_test_subsampled_plus2000.json')]
M='ndcg@5'
for n,a,b in P:
    A=json.load(open(R+a));B=json.load(open(R+b))
    print('\n==',n,A['n_docs'],A['n_queries'],'->',B['n_docs'],B['n_queries'])
    for k in ['ndcg@5','ndcg@10','recall@1','recall@5']:
        print(' base',k,round(A['baseline'][k],3),round(B['baseline'][k],3))
    ra={r['label']:r for r in A['rows']}
    out=[]
    for r in B['rows']:
        o=ra.get(r['label'])
        if not o: continue
        ca=o['retention_ci:'+M];cb=r['retention_ci:'+M]
        out.append((r['compression_vs_float32'],r['group'],r['label'],o['retention:'+M],r['retention:'+M],r['retention:'+M]-o['retention:'+M],(ca[1]-ca[0]+cb[1]-cb[0])/2,o['retention:recall@1'],r['retention:recall@1']))
    out.sort()
    for x in out:
        if abs(x[5])>=0.01 or x[0]>=30 : print(' %6.1fx %-8s %-34s %.3f -> %.3f d=%+.3f ciw=%.3f  r1 %.3f->%.3f'%x)
    print(' n rows',len(out),'mean d',sum(x[5] for x in out)/len(out), 'n |d|>=.03',sum(abs(x[5])>=.03 for x in out))
    # tiered
    ta={(t['hot'],t['cold'],t['candidates']):t for t in A['tiered']}
    for t in B['tiered']:
        o=ta.get((t['hot'],t['cold'],t['candidates']))
        if o: print(' T %-22s %-24s c%d ram%.0fx %.3f->%.3f'%(t['hot'],t['cold'][:24],t['candidates'],t['ram_compression_vs_float32'],o['retention:'+M],t['retention:'+M]))
