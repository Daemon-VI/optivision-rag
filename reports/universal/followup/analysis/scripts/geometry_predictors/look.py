import pandas as pd
pd.set_option('display.width',250);pd.set_option('display.max_columns',50)
df=pd.read_csv('table.csv')
p=df.pivot_table(index=['model','tag'],columns='codec',values='ret')
g=df[df.codec=='int4'].set_index(['model','tag'])[['kind','dim','cos_mean','p2r','p2rc','sm1','sm4','maxabs','qcos','int4_sp','int4m_sp','bin_sp','binm_sp','int4_vc']]
print(g.join(p).round(3).sort_values('int4').to_string())
