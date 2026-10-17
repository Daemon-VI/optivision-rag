from common import *
pd.set_option('display.width',300,'display.max_rows',500,'display.max_columns',40)
w=pp[pp.D>128].copy()
w['col']=w.method.str[0]+w.k.astype(int).astype(str)
t=w.pivot_table(index=['model','ds','D'],columns='col',values='ret').round(3)
print(t.to_string())
n=pp[pp.D<=128]
print(n.groupby(['method','k']).ret.describe().round(3))
# baseline rows for fp16
print(bs[['model','ds','D','baseline','fp16']].query('D>128').to_string())
