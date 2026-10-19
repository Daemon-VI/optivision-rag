from load import *
pd.set_option('display.width',250,'display.max_rows',500,'display.max_colwidth',60)
bs['f']=bs.file.str.replace(R,'',regex=False)
print(bs[['model','ds','D','vpd','base','dtype','nq','nd']].to_string())
