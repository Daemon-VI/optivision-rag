from optivision.adapters import MODELS
import json
for k,m in MODELS.items():
    print(k,'|',m.family,m.modality,m.dimension,m.status,'|',m.evidence[:110],'|',(m.notes or '')[:140])
