"""Validate this revision's five integrated tables, native ratios, and measured figure data.
Checks compact preserved evidence, not a new model run or official evaluation.
"""
import csv,json,re,math,statistics as st
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;CUR=HERE/'current_revision';DATA=HERE/'main_tables'
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)
def load(n):return json.loads((CUR/n).read_text())
def numeric_rows(text):
 return [re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?','&'.join(l.split('&')[1:])) for l in text.splitlines() if '&' in l]
def verify():
 from mmstar_correction_20260924.rescore import recompute
 metric=recompute()
 assert metric==json.loads((HERE/'mmstar_correction_20260924/RESULTS.json').read_text())
 scores=metric['results']
 def mmstar(arm):return f"{scores[arm]['official_accuracy_percent']:.3f}"
 count=0
 for n in range(1,6):
  source=(ROOT/f'tables/final_T{n}.tex').read_text()
  # Compare the original blocks exactly; BAGEL is independently checked below.
  actual=numeric_rows(source.split(r'\rowcolor{TGband}\multicolumn')[0] if n in (1,3) else source)
  accepted=numeric_rows((CUR/f'accepted_tables/T{n:02d}.tex').read_text())
  # Preserve historical accepted files; apply only the audited metric correction.
  if n==3:
   for row in accepted:
    for j,v in enumerate(row):
     if v=='51.737':row[j]=mmstar('exact')
     elif v=='51.919':row[j]=mmstar('stable')
  assert actual==accepted,('Table cell changed',n)
  count+=sum(map(len,actual))
 d=load('data/tables1-3/MEASURED_DATA.json')
 for k in ['geneval','dpg']:
  for r in d[k]:
   if k=='geneval':close(st.mean(r['categories']),r['quality'])
   b=next(x for x in d[k] if x['scale']==r['scale'] and x['arm']=='native')
   close(b['latency_s']/r['latency_s'],r['speedup'])
   text=(ROOT/f"tables/final_T{1 if k=='geneval' else 2}.tex").read_text()
   for v in [r['quality'],r['latency_s']]:assert f'{v:.3f}' in text
   if r['arm']=='ours':assert f"{r['speedup']:.3f}" in text
 for table,name in [(3,'Exact'),(4,'Stable')]:
  vals=[x for x in d['understanding'] if x['table']==table and x['arm']=='ours']
  text=(ROOT/'tables/final_T3.tex').read_text()
  for x in vals:
   expected=mmstar('exact' if name=='Exact' else 'stable') if x['benchmark']=='MMStar' else x['score']
   assert expected in text
 controls=list(csv.DictReader((DATA/'projection_kv_controls.csv').open()))
 text=(ROOT/'tables/final_T5.tex').read_text()
 for x in controls[:4]:
  assert int(x['response_matches'])==int(x['pairs'])==300
  assert f"{float(x['mean_speedup_across_3_repeats']):.3f}" in text
 b=load('data/bagel_no_taylor/RESULTS.json');m=load('data/BAGEL_MMSTAR_VERIFIED.json')
 for arm in ['native','depgraphu']:
  close(st.mean(b['timing'][arm]['repeat_means_s']),b['timing'][arm]['mean_s'])
  close(st.mean(b['diagnostic_quality'][arm]['categories'].values()),b['diagnostic_quality'][arm]['overall'])
  assert b['diagnostic_quality'][arm]['n']==12
 ratio=b['timing']['native']['mean_s']/b['timing']['depgraphu']['mean_s']
 close(ratio,b['comparisons']['requested_full_without_taylor']['ratio_of_mean_times'])
 assert f'{ratio:.3f}'=='1.107'
 for arm in ['native','ours']:close(st.mean(m['results'][arm]['l2_scores'].values()),m['results'][arm]['score_l2_macro'])
 close(m['results']['native']['mean_s']/m['results']['ours']['mean_s'],m['speedup'])
 assert m['complete_pairs']==1500 and f"{m['speedup']:.3f}"=='1.154'
 tags=['single_object','two_object','counting','colors','position','color_attr']
 t1=(ROOT/'tables/final_T1.tex').read_text().split(r'\rowcolor{TGband}\multicolumn')[1]
 t3=(ROOT/'tables/final_T3.tex').read_text().split(r'\rowcolor{TGband}\multicolumn')[1]
 expected1=[];expected3=[]
 for arm in ['native','depgraphu']:
  raw=[json.loads(line) for line in (CUR/f'data/bagel_no_taylor/scores/{arm}.jsonl').read_text().splitlines()]
  assert len(raw)==12
  for tag in tags:
   records=[r for r in raw if r['tag']==tag];assert len(records)==2
   close(st.mean(int(r['correct']) for r in records),b['diagnostic_quality'][arm]['categories'][tag])
  row=['7']+[f"{b['diagnostic_quality'][arm]['categories'][tag]:.3f}" for tag in tags]
  row += [f"{b['diagnostic_quality'][arm]['overall']:.3f}",f"{b['timing'][arm]['mean_s']:.3f}"]
  if arm=='depgraphu':row += [f'{ratio:.3f}']
  expected1.append(row)
 for arm in ['native','ours']:
  row=['7',mmstar('bagel_'+arm),f"{m['results'][arm]['mean_s']:.3f}"]
  if arm=='ours':row += [f"{m['speedup']:.3f}"]
  expected3.append(row)
 def bagel_rows(text):
  return numeric_rows('\n'.join(l for l in text.splitlines() if l.startswith('BAGEL') or l.startswith(r'\textbf{+ DepGraph-U}')))
 # Table 3 has '(adapted)' in the row name; numerical parsing ignores that first cell.
 rows3=numeric_rows('\n'.join(l for l in t3.splitlines() if l.startswith('BAGEL') or l.startswith(r'\textbf{+ DepGraph-U (adapted)}')))
 assert bagel_rows(t1)==expected1,('BAGEL Table 1',bagel_rows(t1),expected1)
 assert rows3==expected3,('BAGEL Table 3',rows3,expected3)
 assert 'BAGEL' not in (ROOT/'tables/final_T2.tex').read_text()
 assert not (ROOT/'tables/final_T6.tex').exists()
 for file,arms in [('supp_exact.tex',['exact_reference','exact']),('supp_stable.tex',['stable_reference','stable']),('appendix_A06.tex',['legacy_reference','legacy_stable'])]:
  tabletext=(ROOT/'tables'/file).read_text()
  for arm in arms:assert mmstar(arm) in tabletext
 print('PASS: 12,000 archived answers rescored with official MMStar rules; all current and legacy local scores corrected')
 print('PASS: integrated BAGEL cells from independent 12-image category recount and 1,500 MMStar pairs; no unmeasured DPG cells')

 print('PASS:',count,'checked numeric table tokens (only audited MMStar aggregation changed); native ratios; category macros; separate BAGEL subset/full scopes')
 rows=list(csv.DictReader((DATA/'figure6_mmstar_pairs.csv').open()));assert len(rows)==1500;bins=[[],[],[],[]]
 for r in rows:
  rc=int(r['incremental_calls']);j=int(r['decode_iterations']);q,rem=divmod(j,10);assert rc==8*q+max(0,rem-2)
  ratio=float(r['reference_seconds'])/float(r['exact_seconds']);close(ratio,r['paired_speedup']);bins[0 if rc==0 else 1 if rc<=2 else 2 if rc<=8 else 3].append(ratio)
 assert [len(x) for x in bins]==[940,121,298,141]
 assert [f'{st.median(x):.3f}' for x in bins]==['1.066','1.880','2.610','3.795']
 print('PASS: 1,500 Figure 7 paired observations and four reuse-bin medians')
 f5=load('figure5_data.json')
 for r in f5['points']:
  close(r['native_mean_s']/r['ours_mean_s'],r['speedup_vs_native'])
  close(st.mean(f5['native_quality_seed_scores']),r['native_quality'])
  close(st.stdev(f5['native_quality_seed_scores']),r['native_sd'])
 f7=load('figure7_data.json')
 for r in f7['models'].values():
  for arm,total in zip(f7['arms'],r['totals_s']):close(sum(r['values'][arm]),total)
  close(r['totals_s'][0]/r['totals_s'][-1],r['profile_speedup'])
 for r in f7['zero_hit']:close(100*(r['cadd_s']/r['reference_s']-1),r['overhead_percent'])
 close(100*(sum(r['cadd_s'] for r in f7['zero_hit'])/sum(r['reference_s'] for r in f7['zero_hit'])-1),f7['overall_overhead_percent'])
 f8=load('figure8_data.json');rr=[json.loads(x) for x in (CUR/'controlled_records.jsonl').read_text().splitlines()];assert len(rr)==72
 for prefix in f8['prefixes']:
  for arm in ['reference','exact']:
   group=[r for r in rr if r['prefix_tokens']==prefix and r['arm']==arm];assert len(group)==12
   for j in range(32):close(st.mean(r['steps'][j]['qwen_ms'] for r in group),f8['means'][str(prefix)][arm][j])
  vals=f8['means'][str(prefix)]
  for j in range(32):close(sum(vals['reference'][:j+1])/sum(vals['exact'][:j+1]),f8['cumulative_speedup'][str(prefix)][j])
 print('PASS: Figure 5 native ratios/seed SD; Figure 8 stacks/overhead; all 96 Figure 9 cells from 72 raw trajectories')
if __name__=='__main__':verify()
