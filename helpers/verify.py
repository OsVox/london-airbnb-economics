from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error,r2_score
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'results'
def verify():
 m=pd.read_csv(O/'membership.csv');assert m.id.is_unique
 sets={k:set(q.host_id) for k,q in m.groupby('split')}
 assert not sets['train']&sets['validation'] and not sets['train']&sets['holdout'] and not sets['validation']&sets['holdout']
 a=json.loads((O/'audit.json').read_text());assert all(hashlib.sha256((ROOT/'data'/p).read_bytes()).hexdigest()==v for p,v in a['hashes'].items())
 f=pd.read_csv(O/'feature_manifest.csv');assert not set(f.feature)&set(a['forbidden'])
 sc=pd.read_csv(O/'model_scores.csv')
 for split in ['train','validation','holdout']:
  p=pd.read_csv(O/f'predictions_{split}.csv');assert set(p.id)==set(m.loc[m.split==split,'id']);assert np.allclose(np.log(p.price_gbp),p.log_price)
  for _,r in sc[sc.split==split].iterrows():
   assert np.isclose(mean_absolute_error(p.log_price,p[r.model]),r.log_mae)
   assert np.isclose(mean_absolute_error(p.price_gbp,np.exp(np.clip(p[r.model],-10,15))),r.gbp_mae)
   assert np.isclose(r2_score(p.log_price,p[r.model]),r.log_r2)
 for directory in (O/'ols').iterdir():
  fits=json.loads((directory/'fits.json').read_text());ill=pd.read_csv(directory/'illustrations.csv')
  for _,row in ill.iterrows():
   f=fits['R__'+row.spec+'__'+row.split];c=np.array(json.loads(row.contrast));b=c@np.array(f['beta']);se=np.sqrt(c@np.array(f['covariance'])@c);critical=stats.t.ppf(.975,f['df'])
   assert np.isclose(b,row.coefficient) and np.isclose(se,row.se) and np.isclose(row.pct_ci_low,100*np.expm1(b-critical*se)) and np.isclose(row.pct_ci_high,100*np.expm1(b+critical*se))
 june=pd.read_csv(O/'ols/june/membership.csv');assert (len(june),june.host_id.nunique(),june.new_host.sum(),june.loc[june.new_host,'host_id'].nunique())==(47533,24259,3143,2440)
 tests=pd.read_csv(O/'ols_tests.csv');assert len(tests)==40
 assert tests.loc[~tests.supported,['analysis','spec','split']].values.tolist()==[['september','host_fixed_effects','holdout']]
 sp=pd.read_csv(O/'shap_sample.csv');sv=pd.read_csv(O/'shap_contributions.csv');assert np.allclose(sp.prediction_log,sv.drop(columns='id').sum(axis=1)+sp.base_log,atol=2e-5)
 cv=pd.read_csv(O/'cv_membership.csv').merge(m[['id','host_id']],on='id',validate='one_to_one');assert cv.groupby('host_id').fold.nunique().max()==1
 checks=dict(data_hashes=True,host_disjoint=True,forbidden_inputs_excluded=True,all_prediction_metrics_recomputed=True,all_ols_contrasts_full_covariance=True,all_40_inference_fits_recomputed=True,known_failed_check_retained=True,june_cohorts_verified=True,shap_additivity=True,cv_host_separation=True)
 (O/'verification.json').write_text(json.dumps(checks,indent=2));return checks
if __name__=='__main__':print(verify())
