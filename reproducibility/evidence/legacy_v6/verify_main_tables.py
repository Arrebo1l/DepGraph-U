"""Check displayed main-table cells and compact recorded observations (standard library)."""
import csv,json,re,math,statistics as st
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;DATA=HERE/'main_tables'
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def load(n):return json.loads((DATA/n).read_text())
def verify():
 d12=load('Tables_1_2_DISPLAYED_DATA.json');d34=load('Tables_3_4_DISPLAYED_DATA.json');d56=load('Tables_5_6_DISPLAYED_DATA.json');nchecked=0
 for n in range(1,7):
  source=(ROOT/f'tables/final_T{n}.tex').read_text().split('\\begin{tabular}',1)[1].split('\\end{tabular}',1)[0];rows=[]
  for line in source.splitlines():
   if '&' in line and any(x in line for x in ('FlashU &','TaskGraph','ML-Cache &','VocabBypass','TaylorSeer &','CADD (Ours)','TGspeed')):
    nums=[]
    for cell in line.split('&')[1:]:nums+=re.findall(r'(?<![A-Za-z])\d+(?:\.\d+)?',cell)
    if nums:rows.append(nums)
  if n in (1,2):expected=[[r['score_display'],r['latency_display'],r['speedup_display']] for r in d12 if r['table']==n]
  elif n in (5,6):expected=[[r['quality'],r['latency'],r['speedup']] for r in d56 if r['table']==n]
  else:
   base=[r for r in d34 if r['table']==n and r['arm']=='baseline'];ours=[r for r in d34 if r['table']==n and r['arm']=='ours']
   expected=[[r['score'] for r in base],[r['score'] for r in ours],[r['latency_display'] for r in base],[r['latency_display'] for r in ours],[r['speedup_display'] for r in ours]]
  assert rows==expected,('Table',n,rows,expected);nchecked+=sum(map(len,expected))
 for r in d12+d34:
  if r['arm']!='ours':continue
  k='scale' if 'scale' in r else 'benchmark';b=next(x for x in d12+d34 if x['table']==r['table'] and x['arm']=='baseline' and x.get(k)==r[k]);close(b['latency_s']/r['latency_s'],r['speedup']);assert f"{r['speedup']:.3f}"==r['speedup_display']
 full=load('Tables_5_6_SOURCE_DATA.json')
 for key,n in [('table5_vocab_bypass',5),('table6_cadd',6)]:
  for bench,x in full[key].items():
   close(x['latency_reference_s']/x['latency_ours_s'],x['speedup']);close(x['quality_reference'],x['quality_ours'])
   for arm,tag in [('reference','reference'),('ours','ours')]:
    row=next(r for r in d56 if r['table']==n and r['benchmark']==bench and r['arm']==arm)
    assert row['latency']==f"{x['latency_'+tag+'_s']:.3f}"
    assert row['quality']==format(x['quality_'+tag],'.5f' if bench=='GenEval' else '.3f')
    assert row['speedup']==f"{x['speedup'] if arm=='ours' else 1:.3f}"
 controls=list(csv.DictReader((DATA/'projection_kv_controls.csv').open()))
 text=(ROOT/'tables/final_T3.tex').read_text()
 for c in controls[:4]:
  assert int(c['response_matches'])==int(c['pairs'])==300
  assert f"{float(c['mean_speedup_across_3_repeats']):.3f}" in text
  assert f"{float(c['sd']):.3f}" in text
 print('PASS: %d current main-table values; unrounded paired ratios; retained projection/KV controls'%nchecked)
 rows=list(csv.DictReader((DATA/'figure6_mmstar_pairs.csv').open()));assert len(rows)==1500;bins=[[],[],[],[]]
 for r in rows:
  rc=int(r['incremental_calls']);j=int(r['decode_iterations']);q,rem=divmod(j,10);assert rc==8*q+max(0,rem-2)
  ratio=float(r['reference_seconds'])/float(r['exact_seconds']);close(ratio,r['paired_speedup']);bins[0 if rc==0 else 1 if rc<=2 else 2 if rc<=8 else 3].append(ratio)
 assert [len(x) for x in bins]==[940,121,298,141]
 assert [f'{st.median(x):.3f}' for x in bins]==['1.066','1.880','2.610','3.795']
 a=[float(r['reference_seconds']) for r in rows];b=[float(r['exact_seconds']) for r in rows]
 close(st.mean(a),2.5408948834910987);close(st.mean(b),1.0867759574825564)
 assert f'{st.median(a)/st.median(b):.3f}'=='1.076'
 def p95(v):
  v=sorted(v);x=(len(v)-1)*.95;i=int(x);return v[i]+(v[i+1]-v[i])*(x-i)
 assert f'{p95(a)/p95(b):.3f}'=='4.115'
 print('PASS: all 1,500 Figure 6 paired observations, reuse bins and dataset-statistic ratios')
if __name__=='__main__':verify()
