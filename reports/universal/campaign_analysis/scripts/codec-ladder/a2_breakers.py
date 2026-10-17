import numpy as np, pandas as pd
from prep import *
pd.set_option('display.width',300); pd.set_option('display.max_rows',200); pd.set_option('display.max_colwidth',45)
w=pd.read_csv('wide_ret.csv')
cols=['model','ds','dim','vpd','fp16']+CODECS
print("== int4 (uncentred) < 0.97"); print(w[w.int4<0.97][cols].round(4).sort_values('int4').to_string())
print("== int4(mean) < 0.99"); print(w[w['int4(mean)']<0.99][cols].round(4).to_string())
print("== lloyd2 < binary - 0.005"); print(w[w['lloyd2(7)']<w.binary-0.005][cols].round(4).to_string())
print("== int4 < lloyd2 - 0.005 but int4 >= .97"); print(w[(w.int4<w['lloyd2(7)']-0.005)&(w.int4>=0.97)][cols].round(4).to_string())
print("== lloyd2<0.95"); print(w[w['lloyd2(7)']<0.95][cols].round(4).to_string())
print("== int8<0.997"); print(w[w['int8(per_vector)']<0.997][cols].round(4).to_string())
# int4 collapse by model: both datasets?
c=w.assign(col=w.int4<0.97).groupby('model').col.agg(['sum','count']); print(c[c['sum']>0])
