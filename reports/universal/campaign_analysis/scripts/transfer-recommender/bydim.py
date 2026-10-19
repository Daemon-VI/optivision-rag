from lomo import *
import pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
dim=np.array([u['dim'] for u in U])
for pk in ['hm0.25+pca0.25+f32','hm0.25+pca0.125+f32','pca0.5+f32','hm0.25+bin','bin','hm0.25+int4m','lloyd2','hm0.33+bin']:
    j=[key_str(k) for k in CAND].index(pk)
    print('==',pk)
    for d_ in sorted(set(dim)):
        a=(dim==d_)&~np.isnan(R[:,j])
        if a.any(): print(f'  dim {d_:5d} mod {mod[a][0]:5s} n={a.sum():3d} ret med {np.median(R[a,j]):.3f} min {R[a,j].min():.3f} max {R[a,j].max():.3f} comp {np.median(C[a,j]):.0f}  pass95 {np.mean(R[a,j]>=.95):.2f} pass97 {np.mean(R[a,j]>=.97):.2f}')
