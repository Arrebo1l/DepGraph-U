#!/usr/bin/env python3
"""Recompute compact published evidence; does not run models or evaluators."""
import csv
import hashlib
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

def load(name):
    return json.loads((HERE / name).read_text())

def rows(name):
    with (HERE/name).open(newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(a, b, rel=1e-9, abs_tol=1e-7):
    assert math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol), (a, b)

def main():
    manifest=load('MANIFEST.json')
    for name,digest in manifest['files'].items():
        assert sha(HERE/name)==digest, name
    fig=load('figure_provenance.json')
    for r in fig['files']:
        assert sha(HERE.parent/r['file'])==r['sha256'], r['file']
    print('PASS: %d evidence hashes and %d native figure hashes' % (len(manifest['files']),len(fig['files'])))

    data=load('table_data.json')
    for m,target in enumerate(data['A4_macro']):
        close(st.mean(c['values'][m][0]/c['values'][m][1] for c in data['A4_counts']), target)
    assert sum(int(r['n']) for r in data['A5_exact'])==24769
    token_n=0
    for r in data['A5_exact']:
        assert int(r['response_matches'])==int(r['response_pairs'])==int(r['n'])
        close(float(r['flashu_mean_s'])/float(r['exact_mean_s']), r['mean_speedup'])
        if r['token_pairs_if_available']!='NA':
            assert r['token_pairs_if_available']==r['token_matches_if_available']
            token_n+=int(r['token_pairs_if_available'])
    assert token_n==20781
    for r in data['A6_timing']:
        close(float(r['paired_flashu_mean_s'])/float(r['stable_mean_s']),r['mean_speedup'])
    assert sum(int(r['response_pairs'])-int(r['response_matches']) for r in data['A6_timing'])==1998
    control=data['A7_standard']
    close(control['no_kv_timing']['mean_s']/control['with_kv_timing']['mean_s'],control['mean_speedup'])
    close(control['independent_official_rule_recount_score'],433/900)
    assert control['sample_equality']['identical_responses']==900
    print('PASS: category macros, six matched latency ratios, 24,769 response matches, 20,781 token matches, native KV control')

    traces=rows('trace_examples.csv'); by_task=defaultdict(list);by_len=defaultdict(list)
    for r in traces:
        j=int(r['generated_tokens']); q,rem=divmod(j,10)
        assert int(r['rebuild_calls'])==2*q+min(rem,2)
        assert int(r['incremental_calls'])==8*q+max(0,rem-2)
        close(float(r['rebuild_ms'])+float(r['incremental_ms']),r['traced_ms'])
        by_task[r['benchmark']].append(r)
        group='2' if j==2 else '3--9' if j<10 else '10--19' if j<20 else '20--49' if j<50 else '50--128'
        by_len[group].append(r)
    assert len(traces)==24769
    assert sum(int(r['generated_tokens']) for r in traces)==65828
    for key,rr in by_task.items():
        expected=data['A8_recomputed'][key]
        assert len(rr)==expected['examples']
        assert sum(int(r['incremental_calls'])==0 for r in rr)==expected['zero_incremental_examples']
        assert sum(int(r['generated_tokens']) for r in rr)==expected['decode_steps']
    for target in data['A8_length_timing']:
        rr=by_len[target['length'].replace('–','--').replace('-', '--') if '--' not in target['length'] else target['length']]
        assert len(rr)==target['n']
        rb=sum(float(r['rebuild_ms']) for r in rr); inc=sum(float(r['incremental_ms']) for r in rr)
        calls=sum(int(r['incremental_calls']) for r in rr)
        close(st.mean(float(r['lm_generation_ms']) for r in rr)/1000,target['lm_generation_s'])
        close(rb/len(rr),target['rebuild_ms_per_example'])
        close(100*rb/(rb+inc),target['rebuild_share_percent'])
        if calls: close(inc/calls,target['incremental_ms_per_call'])
        else: assert target['incremental_ms_per_call'] is None
    print('PASS: all 65,828 call counts and five observed-length timing groups')

    seeds=rows('interval_seed_details.csv'); assert len(seeds)==30
    for target in data['A9_summary']:
        for arm in ['original','full']:
            rr=[r for r in seeds if r['interval']==target['interval'] and r['arm']==arm]
            assert len(rr)==3 and sum(int(r['n_images']) for r in rr)==360
            close(st.mean(float(r['mean_latency_s']) for r in rr),target[arm+'_mean_s'])
            close(st.stdev(float(r['mean_latency_s']) for r in rr),target[arm+'_latency_seed_sd'])
            close(st.mean(float(r['geneval_subset_score']) for r in rr),target[arm+'_subset_quality'])
            close(st.stdev(float(r['geneval_subset_score']) for r in rr),target['quality_seed_sd'])
        close(float(target['original_mean_s'])/float(target['full_mean_s']),target['speedup'])
        assert target['equal_pairs']==target['total_pairs']=='360'
    print('PASS: five intervals, three seeds, both arms, sample standard deviations')

    cases=load('case_records.json'); assert len(cases)==4
    for c in cases:
        for arm,img in c['images'].items():
            assert sha(HERE/img['file'])==img['sha256']==c['paired_sha256']
        assert c['outcomes']['original']['correct']==c['outcomes']['full']['correct']
    outcomes=rows('geneval_paired_outcomes.csv')
    assert len(outcomes)==2212
    passed=lambda r: r['official_correct'].lower() in ['true','1']
    assert sum(map(passed,outcomes))==1518
    cat=defaultdict(list)
    for r in outcomes:cat[r['tag']].append(passed(r))
    close(st.mean(st.mean(v) for v in cat.values()),data['A4_macro'][1])
    print('PASS: four historical fixed-policy image pairs and 2,212 official paired outcomes')

    native=load('current_revision/native_cases/SELECTED_CASES.json')
    assert native['model']=='Show-o2-1.5B' and native['seed']==0
    assert native['native_timepoints']==100 and native['full_system_timepoints']==70
    assert [c['index'] for c in native['cases']]==[451,514,548,506]
    correct=0
    for c in native['cases']:
        assert c['prompt'] and c['native_and_system_outputs_differ']
        for arm,img in c['images'].items():
            assert sha(HERE/img['file'])==img['sha256']
            assert img['width']==img['height']==432
            correct+=int(img['gen_eval_correct'])
        assert c['images']['showo']['pixel_sha256']!=c['images']['full']['pixel_sha256']
    assert correct==7
    for f in load('current_revision/APPENDIX_FIGURE_MANIFEST.json'):
        assert sha(HERE.parent/f['pptx'])==f['pptx_sha256']
        assert sha(HERE.parent/f['pdf'])==f['pdf_sha256']
    print('PASS: four distinct native/system image pairs, seven recorded passes, four editable appendix figure sources')

    routes=load('uqp_frozen_routes.json'); counts=Counter()
    for r in routes['rows']:
        feasible=[k for k,v in r['candidate_scores'].items() if v>=r['baseline_dpg_score']]
        chosen=min(feasible,key=lambda k:routes['latency_constants_seconds'][k])
        assert chosen==r['selected_route'];counts[chosen]+=1
    assert dict(counts)==routes['counts']
    pairs=rows('uqp_paired_records.csv'); assert len(pairs)==1065
    for seed,expected in [(0,(0,867,198)),(1,(254,683,128))]:
        ds=[float(r['delta_dpg_seed'+str(seed)]) for r in pairs]
        assert (sum(v < -1e-8 for v in ds),sum(abs(v)<=1e-8 for v in ds),sum(v>1e-8 for v in ds))==expected
        if seed==1:close(st.mean(ds),data['A10_seed1']['seed1_quality_drop_points'])
    timed=[r for r in pairs if r['reference_seconds'] and r['frozen_seconds']]
    assert len(timed)==1064
    ref=st.mean(float(r['reference_seconds']) for r in timed)
    frozen=st.mean(float(r['frozen_seconds']) for r in timed)
    close(ref,data['A10_seed1']['seed1_measured_35_step_reference_with_cadd_vocab_seconds'])
    close(frozen,data['A10_seed1']['seed1_measured_routed_seconds'])
    close(ref/frozen,data['A10_seed1']['seed1_ratio_vs_matched_execution_reference'])
    print('PASS: all frozen route assignments, two quality-difference distributions and 1,064 matched timing pairs')
    from verify_main_tables import verify
    verify()
    print('All compact evidence checks passed. No GPU inference; MMStar archived-answer rescoring is included.')

if __name__=='__main__':
    main()
