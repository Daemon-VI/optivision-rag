import json
d=json.load(open('text/text-answerai-colbert-small-v1-scifact.json'))
for r in d['rows']:
    print(r['label'],r.get('retention:ndcg@5'),r.get('retention_ci:ndcg@5'))
