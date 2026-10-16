"""Cross-artifact checks, plus read-only preservation verification."""
from pathlib import Path
import json,hashlib,zipfile,xml.etree.ElementTree as E,re
import pandas as pd
import nbformat
from verify import verify
R=Path(__file__).resolve().parents[1];O=R/'results'
checks=verify();nb=nbformat.read(R/'london_airbnb_group_project.ipynb',4);code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert [c.execution_count for c in code]==list(range(1,len(code)+1))
assert not [out for c in code for out in c.outputs if out.output_type=='error']
assert sum('image/png' in out.get('data',{}) for c in code for out in c.outputs)>=9
z=zipfile.ZipFile(R/'london_airbnb_group_project.pptx');ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
slides=[p for p in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+.xml',p)];notes=[p for p in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',p)];charts=[p for p in z.namelist() if re.fullmatch(r'ppt/slides/charts/chart\d+.xml',p)]
assert len(slides)==54 and len(notes)==54 and len(charts)==10
text='\n'.join(' '.join(E.fromstring(z.read(p)).itertext()) for p in slides)
assert all(n in text for n in ['57.10','62.97','5.2%','11.2%','0.317','8.33%','8.96%','59','635'])
for p in notes:
 root=E.fromstring(z.read(p));body=' '.join(t.text or '' for t in root.findall('.//a:t',ns));assert len(body)>250
# Native chart caches correspond to the executed CSVs, up to declared rounding.
sc=pd.read_csv(O/'model_scores.csv');expected=sc.query("split=='holdout'").set_index('model');cache=[]
for p in charts:
 root=E.fromstring(z.read(p));cache.append([float(t.text) for t in root.findall('.//c:val/c:numLit/c:pt/c:v',ns)]+[float(t.text) for t in root.findall('.//c:val/c:numRef/c:numCache/c:pt/c:v',ns)])
assert any(len(v)==6 and all(abs(a-b)<1e-9 for a,b in zip(v,[expected.loc[n,'log_mae'] for n in ['Median','OLS','Ridge','Small tree','Random forest','XGBoost']])) for v in cache)
# Scatter caches preserve all 802 observed-rating cases and exact grouped values.
import numpy as np
sample=pd.read_csv(O/'shap_sample.csv');contribution=pd.read_csv(O/'shap_contributions.csv')
valid=sample.review_scores_rating.notna()
groups9=[sample.number_of_reviews<=10,(sample.number_of_reviews>10)&(sample.number_of_reviews<=100),sample.number_of_reviews>100]
groups10=[sample.review_scores_rating<4.7,(sample.review_scores_rating>=4.7)&(sample.review_scores_rating<4.9),sample.review_scores_rating>=4.9]
for chart_no,groups,field in [(9,groups9,'review_scores_rating'),(10,groups10,'number_of_reviews')]:
 root=E.fromstring(z.read(f'ppt/slides/charts/chart{chart_no}.xml'));series=root.findall('.//c:ser',ns)
 assert len(series)==3
 count=0
 for group,ser in zip(groups,series):
  selected=valid&group;expected_x=sample.loc[selected,field].to_numpy()
  if chart_no==10:expected_x=np.log1p(expected_x)
  expected_y=contribution.loc[selected,field].to_numpy()
  x=[float(t.text) for t in ser.findall('.//c:xVal//c:pt/c:v',ns)]
  y=[float(t.text) for t in ser.findall('.//c:yVal//c:pt/c:v',ns)]
  assert len(x)==len(y)==int(selected.sum())
  assert np.allclose(x,expected_x,rtol=1e-8,atol=1e-10) and np.allclose(y,expected_y,rtol=1e-8,atol=1e-10)
  count+=len(x)
 assert count==int(valid.sum())==802
checks['native_scatter_data_verified']=True
receipt=json.loads((R/'.build/final-validation-v3.json').read_text());assert receipt['finalSha256']==hashlib.sha256((R/'london_airbnb_group_project.pptx').read_bytes()).hexdigest()
assert receipt['firstPartyImport']['passed'] and receipt['presentationLayout']['finding_count']==0
assert len(list((R/'.build/pdf-rendered').glob('slide-*.png')))==54
assert (R/'london_airbnb_group_project.pdf').stat().st_size>100000
# Existing artifacts were never modified by this submission.
old=json.loads((R/'provenance/original_files_sha256.json').read_text());changed=[]
for p,h in old.items():
 f=R.parent/p
 if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:changed.append(p)
checks.update(clean_kernel_execution_counts=True,notebook_error_free=True,visible_figures=True,native_pptx_charts=10,pptx_slides=54,pptx_notes=54,main_slides=18,appendix_slides=36,chart_scores_agree=True,final_pptx_hash_verified=True,all_pdf_pages_rendered=True,original_file_changes=changed)
assert not changed,changed
(O/'completion_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
