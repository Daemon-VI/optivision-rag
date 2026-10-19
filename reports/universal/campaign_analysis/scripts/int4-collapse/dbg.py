import pandas as pd
d=pd.read_pickle('master2.pkl')
print(d[d.src=='wide'][['model','dataset']].head(5)); print(d[d.src!='wide'][['model','dataset']].head(5))
