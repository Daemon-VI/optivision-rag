from optivision.adapters import MODELS
for k,v in sorted(MODELS.items()):
    print(k,'|',v.family,'|',v.modality,v.dimension,'|',v.evidence[:70] if v.evidence else '')
