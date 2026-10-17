from optivision.adapters import MODELS
for k,v in MODELS.items(): print(k,'|',v.family,v.modality,v.dimension,v.status,'|',v.evidence[:160],'|',(v.notes or '')[:100])
