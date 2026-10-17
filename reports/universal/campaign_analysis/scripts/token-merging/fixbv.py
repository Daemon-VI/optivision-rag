import re
s=open('build.py').read()
s=s.replace("""        m=j['baseline'].get('encoder',{}).get('model') if isinstance(j['baseline'].get('encoder'),dict) else None
        ds=j['dataset'].split('/')[-1].split('_')[0]
        if m: bv[(m,ds)]=j['baseline']['vectors_per_doc']""","""        ds=j['dataset'].split('/')[-1].split('_')[0]
        parts=f.replace('\\','/').split('/')
        if 'campaign' in parts:
            m=parts[parts.index('campaign')+1].replace('--','/',1)
            bv[(m,ds)]=j['baseline']['vectors_per_doc']""")
open('build.py','w').write(s)
