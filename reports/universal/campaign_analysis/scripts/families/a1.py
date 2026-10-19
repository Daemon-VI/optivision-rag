from common import *
import json
x=M["vidore/colpali-v1.3-merged"]["vidore/docvqa_test_subsampled"]
print(x.get("source"),x["note"] if "note" in x else None)
print([r["label"] for r in x["rows"]][:70])
x=M["vidore/colpali-v1.2"]["vidore/docvqa_test_subsampled"]; print(x["source"])
y={r['label']:r['retention'] for r in x['rows']}
print(y['hierarchical_merge(0.33) > project(128)'], y['hierarchical_merge(0.33)'])
print(M["vidore/colpali-v1.2"]["vidore/docvqa_test_subsampled"]["rows"][8])
