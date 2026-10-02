"""Print result tables from the published JSON, without hard-coded measurements."""
from pathlib import Path
import json
from artifact_utils import ratio_of_means

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT/'results/summary.json').read_text())
    print('Fixed-policy Show-o2 generation (NOT native-model denominator)')
    print('| Benchmark | Scale | Reference s | Ours s | Speedup | Quality ref → ours |')
    print('|---|---|---:|---:|---:|---:|')
    rows = data['fixed_policy_generation']
    for a in rows:
        if a['arm'] != 'baseline': continue
        b = next(r for r in rows if r['arm']=='ours' and r['table']==a['table'] and r['scale']==a['scale'])
        speed = ratio_of_means([a['latency_s']], [b['latency_s']])
        name = {1:'GenEval', 2:'DPG-Bench'}[a['table']]
        print(f'| {name} | {a["scale"]} | {a["latency_s"]:.3f} | {b["latency_s"]:.3f} | {speed:.3f}× | {a["score"]:.5f} → {b["score"]:.5f} |')
    print('\nExact-DLS + last-position projection vs. reconstructed FlashU')
    print('| Benchmark | Reference s | Ours s | Speedup | Quality ref → ours |')
    print('|---|---:|---:|---:|---:|')
    rows = [r for r in data['understanding'] if r['table']==3]
    for a in rows:
        if a['arm'] != 'baseline': continue
        b=next(r for r in rows if r['arm']=='ours' and r['benchmark']==a['benchmark'])
        speed=ratio_of_means([a['latency_s']],[b['latency_s']])
        print(f'| {a["benchmark"]} | {a["latency_s"]:.3f} | {b["latency_s"]:.3f} | {speed:.3f}× | {float(a["score"]):.3f} → {float(b["score"]):.3f} |')
    print('\nMMStar uses corrected overall accuracy; see docs/errata.md.')
    print('MME quality is perception score; its timing includes P+C.')


if __name__ == '__main__':
    main()
