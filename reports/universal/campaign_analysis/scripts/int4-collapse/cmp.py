import pandas as pd
w=pd.read_pickle('wide.pkl'); m=pd.read_pickle('master.pkl')
print(w.model.isna().sum()); print(w[w.model.isna()].file.tolist()[:10])
print(w[['file','dim','vpd','base','enc_dtype']].head(10).to_string())
print(m.columns.tolist())
