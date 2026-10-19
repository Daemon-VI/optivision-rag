from common import *
import collections
c=collections.Counter((SHORT[ds],x['metric']) for m,v in M.items() for ds,x in v.items()); print(c)
