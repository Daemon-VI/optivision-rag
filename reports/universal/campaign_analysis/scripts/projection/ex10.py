import json,glob,os
for f in glob.glob('geometry/*.json'):
    d=json.load(open(f));print(os.path.basename(f)[:40],round(d['median_nn_cosine_within_doc'],3),round(d['corpus_mean_norm'],3),round(d['query_cosine_to_corpus_mean_median'],3))
