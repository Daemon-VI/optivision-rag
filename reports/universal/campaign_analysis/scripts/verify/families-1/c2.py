from optivision.adapters import MODELS
for k in ["nvidia/nemotron-colembed-vl-4b-v2","nvidia/nemotron-colembed-vl-8b-v2","mixedbread-ai/mxbai-edge-colbert-v0-17m","lightonai/LateOn","lightonai/mLateOn","lightonai/GTE-ModernColBERT-v1","lightonai/ColBERT-Zero","TomoroAI/tomoro-colqwen3-embed-4b","webAI-Official/webAI-ColVec1-4b"]:
    v=MODELS[k]; print(k,'\n ',v.evidence,'\n ',v.notes,'\n')
