import json,glob,os
def L(p):
    try: return json.load(open(p))
    except: return None
R={}
for f in glob.glob('f*/*/geometry/*.json'):
    g=json.load(open(f)); parts=f.replace(chr(92),'/').split('/'); tag=parts[-1][:-5]
    R[(parts[-3],tag)]=g
print('model/tag | pearson int4 | int4mean | cos_mean | p2r | p2rc | top1 | maxabs | qcos | nvec')
for (m,t),g in sorted(R.items(),key=lambda x:x[0][1].split('_')[-1]):
    if 'scifact' not in t and 'nfcorpus' not in t: continue
    c=g['codecs']
    print(t, round(c['int4']['score_pearson']['mean'],3), round(c['int4(mean)']['score_pearson']['mean'],3), round(g['doc_cos_to_mean']['mean'],3), round(g['peak_to_rms']['mean'],2), round(g['peak_to_rms_centred']['mean'],2), round(g['second_moment_share']['top1'],3), round(g.get('max_abs_coordinate_mean',0),3), round(g['query_cos_to_doc_mean']['mean'],3), g['dim'], g.get('n_docs'),g.get('sample'))
