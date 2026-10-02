"""Offline application of MMStar's published answer rule and overall aggregation.
No inference, answer rewriting, or metric selection; every archived answer is retained.
Rule: https://github.com/MMStar-Benchmark/MMStar/blob/main/eval/vlmeval/evaluate/mmstar.py
Lines 49--64 (answer matching), 66--76 (overall correct / 1500).
"""
import json,collections,statistics,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent

def correct(response, answer):
    answer=str(answer).lower().strip().replace('\n',' ')
    predict=str(response).lower().strip().replace('\n',' ')
    try:
        if answer == predict[0]: return 1
        elif predict[0] == '(' and answer == predict[1]: return 1
        elif predict[0:7] == 'option ' and answer == predict[7]: return 1
        elif predict[0:14] == 'the answer is ' and answer == predict[14]: return 1
    except IndexError: pass
    return 0

def recompute():
    records=[json.loads(l) for l in (HERE/'responses.jsonl').read_text().splitlines()]
    result={}
    for arm in sorted({r['arm'] for r in records}):
        rows=[r for r in records if r['arm']==arm]
        assert len(rows)==1500 and sorted(r['index'] for r in rows)==list(range(1500))
        assert sorted(collections.Counter(r['category'] for r in rows).values())==[250]*6
        groups=collections.defaultdict(list)
        for r in rows:groups[r['l2_category']].append(r['saved_correct'])
        assert len(groups)==18
        scores=[correct(r['response'],r['answer']) for r in rows]
        result[arm]={'n':1500,'correct':sum(scores),'official_accuracy_percent':sum(scores)/15,
            'historical_l2_macro_percent':100*statistics.mean(statistics.mean(x) for x in groups.values()),
            'answer_rule_disagreements':sum(a!=r['saved_correct'] for a,r in zip(scores,rows))}
    byarm={a:{r['index']:r for r in records if r['arm']==a} for a in result}
    assert all(byarm['exact'][i]['response']==byarm['exact_reference'][i]['response'] for i in range(1500))
    return {'method':'Published MMStar answer rule; overall correct / 1500, expressed as percent',
        'source':'https://github.com/MMStar-Benchmark/MMStar/blob/main/eval/vlmeval/evaluate/mmstar.py',
        'responses_sha256':hashlib.sha256((HERE/'responses.jsonl').read_bytes()).hexdigest(),
        'results':result}

if __name__=='__main__':
    actual=recompute(); p=HERE/'RESULTS.json'
    if p.exists():assert actual==json.loads(p.read_text())
    else:p.write_text(json.dumps(actual,indent=2)+'\n')
    print(json.dumps(actual,indent=2))
