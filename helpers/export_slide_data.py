from pathlib import Path
import json
import pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'results'
names=['model_scores','ols_tests','ols_contrasts','ols_focus_coefficients','ols_full_training_coefficients','hist_price','hist_log_price','hist_rating','hist_reviews','location_medians','response_contrasts','response_support','response_grid','matched_ablation','development_cv','permutation_summary','paired_host_bootstrap','error_audit','tuning','local_examples','shap_sample','shap_contributions','shap_importance']
data={name:pd.read_csv(O/f'{name}.csv').fillna('').to_dict('records') for name in names}
for name in ['audit','selected_settings','ols_full_design_note','verification']:data[name]=json.loads((O/f'{name}.json').read_text())
(R/'.build/slide_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
