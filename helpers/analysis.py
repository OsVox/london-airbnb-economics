"""Reproducible analysis. All paths relative to this submission; no network at fitting time."""
from pathlib import Path
import json, hashlib, subprocess, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import sparse, linalg, stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor, DMatrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from .features import prepare
from .ols_core import contrast
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'; FIG=ROOT/'figures'; SEED=20261016
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.bbox':'tight'})
COLORS=['#176B82','#E28B36','#587255','#6C5592']

def save_json(name, value):
 (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
def savefig(name):
 plt.savefig(FIG/f'{name}.png');plt.close()
def metrics(y,p):
 return dict(log_mae=mean_absolute_error(y,p),gbp_mae=mean_absolute_error(np.exp(y),np.exp(np.clip(p,-10,15))),log_r2=r2_score(y,p))
def numeric_price(d):return pd.to_numeric(d.price.astype(str).str.replace(r'[^0-9.\-]','',regex=True),errors='coerce')
def preprocess(X,num,cat):
 return ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),num),('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=20,drop='first'),cat)],sparse_threshold=1.)
def dense(x):return x.toarray() if sparse.issparse(x) else np.asarray(x)
def xgb(depth=5):return XGBRegressor(max_depth=depth,n_estimators=300,learning_rate=.05,min_child_weight=20,subsample=.8,colsample_bytree=.8,reg_lambda=10,tree_method='hist',n_jobs=4,random_state=SEED,objective='reg:squarederror')
def audit():
 raw=pd.read_csv(ROOT/'data/london_2026-09-17.csv.gz',low_memory=False);p=numeric_price(raw)
 valid=p.gt(0)&raw.host_id.notna()&raw.latitude.between(-90,90)&raw.longitude.between(-180,180)&raw.accommodates.ge(1)
 use=valid&p.between(10,2000);d=raw.loc[use].copy().reset_index(drop=True)
 split=pd.read_csv(ROOT/'data/split.csv');assert set(d.id)==set(split.id)
 d=d.merge(split,on=['id','host_id'],validate='one_to_one',sort=False);d['price_gbp']=numeric_price(d);y=np.log(d.price_gbp.to_numpy());X,num,cat=prepare(d);num+=['distance_km','latitude','longitude'];cat+=['neighbourhood_cleansed']
 # Recreate the original split, including ordering.
 tr,rest=next(GroupShuffleSplit(n_splits=1,test_size=.4,random_state=SEED).split(X,groups=d.host_id));v,t=next(GroupShuffleSplit(n_splits=1,test_size=.5,random_state=SEED).split(X.iloc[rest],groups=d.iloc[rest].host_id))
 ix={'train':tr,'validation':rest[v],'holdout':rest[t]}
 for k,indices in ix.items():assert d.iloc[indices].split.eq(k).all()
 for a,b in [('train','validation'),('train','holdout'),('validation','holdout')]:assert not set(d.iloc[ix[a]].host_id)&set(d.iloc[ix[b]].host_id)
 forbidden=['price','price_quote_raw','price_quote_total_price','price_quote_price_per_night','review_scores_value','review_scores_location','host_id','id','estimated_revenue_l365d','estimated_occupancy_l365d']
 assert not set(forbidden)&set(num+cat)
 # Source payloads are used ONLY to audit the target currency, never as model input.
 currency=raw.price_quote_raw.fillna('').str.extract(r'"currency"\s*:\s*"([^\"]+)"',expand=False).value_counts().to_dict()
 quotes=pd.to_numeric(raw.price_quote_price_per_night,errors='coerce');both=p.notna()&quotes.notna();matches=np.isclose(p[both],quotes[both],atol=.011)
 raw.isna().mean().rename('missing_fraction').sort_values(ascending=False).to_csv(OUT/'raw_missingness.csv')
 flow=pd.DataFrame([['Raw September',len(raw),raw.host_id.nunique()],['Positive price + basic validity',int(valid.sum()),raw.loc[valid,'host_id'].nunique()],['Price £10–£2000',len(d),d.host_id.nunique()]],columns=['cohort','listings','hosts'])
 eligible=d.room_type.isin(['Entire home/apt','Private room'])&d.review_scores_rating.between(3,5)&d.number_of_reviews.between(1,500)
 flow.loc[len(flow)]=['Rated hypothesis cohort',int(eligible.sum()),d.loc[eligible,'host_id'].nunique()];flow.to_csv(OUT/'cohort_flow.csv',index=False)
 d[['id','host_id','split','price_gbp']].assign(hypothesis_eligible=eligible).to_csv(OUT/'membership.csv',index=False)
 mf=pd.DataFrame({'feature':num+cat});mf['unique_values']=mf.feature.map(X.nunique());mf['missing_fraction']=mf.feature.map(X.isna().mean());mf['group']=mf.feature.map(feature_group);mf.to_csv(OUT/'feature_manifest.csv',index=False)
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*')}
 a={'raw_rows':len(raw),'raw_columns':raw.shape[1],'raw_hosts':raw.host_id.nunique(),'duplicates':int(raw.id.duplicated().sum()),'missing_prices':int(p.isna().sum()),'all_empty_columns':int(raw.isna().all().sum()),'currency_counts':currency,'quote_pairs':int(both.sum()),'quote_matches':int(matches.sum()),'cohort':len(d),'hosts':d.host_id.nunique(),'splits':{k:{'rows':len(i),'hosts':d.iloc[i].host_id.nunique()} for k,i in ix.items()},'hypothesis_rows':int(eligible.sum()),'hypothesis_hosts':d.loc[eligible,'host_id'].nunique(),'scrape_dates':raw.last_scraped.value_counts().to_dict(),'hashes':hashes,'forbidden':forbidden}
 assert a['raw_rows']==91954 and a['raw_columns']==90 and a['cohort']==59635 and a['hypothesis_rows']==46553
 save_json('audit.json',a)
 return d,X,y,num,cat,ix,a

def feature_group(c):
 if c in ['distance_km','latitude','longitude','neighbourhood_cleansed']:return 'Location'
 if c.startswith('amenity_'):return 'Amenities'
 if c.startswith('review') or c=='number_of_reviews':return 'Reviews'
 if c.startswith('host_'):return 'Host'
 if c.startswith('quote_') or c in ['scrape_day','minimum_nights','maximum_nights','instant_bookable']:return 'Booking context'
 return 'Property'

def eda(d,X,ix):
 t=d.iloc[ix['train']].copy();t['distance_km']=X.iloc[ix['train']].distance_km.to_numpy()
 fig,axs=plt.subplots(1,2,figsize=(11,3.7))
 for ax,values,name in [(axs[0],t.price_gbp,'Цена предложения, £'),(axs[1],np.log(t.price_gbp),'Натуральный логарифм цены')]:
  counts,bins=np.histogram(values,bins=35);ax.stairs(counts,bins,fill=True,color=COLORS[0]);ax.set(xlabel=name,ylabel='Объявления');pd.DataFrame({'left':bins[:-1],'right':bins[1:],'count':counts}).to_csv(OUT/('hist_log_price.csv' if ax==axs[1] else 'hist_price.csv'),index=False)
 fig.suptitle('Обучающая выборка: правый хвост цены сжимается после логарифмирования');fig.tight_layout();savefig('price_distributions')
 fig,axs=plt.subplots(1,2,figsize=(11,3.7));axs[0].hist(t.review_scores_rating.dropna(),bins=25,color=COLORS[0]);axs[0].set(xlabel='Средний рейтинг',ylabel='Объявления');axs[1].hist(np.log1p(t.number_of_reviews),bins=30,color=COLORS[1]);axs[1].set(xlabel='log(1 + число отзывов)',ylabel='Объявления');fig.tight_layout();savefig('review_distributions')
 for col,name in [('review_scores_rating','rating'),('number_of_reviews','reviews')]:
  v=t[col].dropna();v=np.log1p(v) if name=='reviews' else v;c,b=np.histogram(v,25);pd.DataFrame({'left':b[:-1],'right':b[1:],'count':c}).to_csv(OUT/f'hist_{name}.csv',index=False)
 t['distance_band']=pd.cut(t.distance_km,[0,3,6,10,15,1000],labels=['0–3','3–6','6–10','10–15','15+'],include_lowest=True)
 loc=t[t.room_type.isin(['Entire home/apt','Private room'])].groupby(['room_type','distance_band'],observed=True).agg(n=('id','size'),median_price=('price_gbp','median')).reset_index();loc.to_csv(OUT/'location_medians.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,4))
 for c,(label,g) in zip(COLORS,loc.groupby('room_type')):ax.plot(g.distance_band.astype(str),g.median_price,'o-',label=label,color=c)
 ax.set(xlabel='Расстояние до фиксированного центра, км',ylabel='Медиана цены, £',title='Описательные медианы: различия качества не учтены');ax.legend();fig.tight_layout();savefig('location_medians')
 return loc

def inference():
 for script in ['analyse.py','replicate.py','shape_september.py','shape_june.py']:
  start=time.time();res=subprocess.run([sys.executable,str(ROOT/'helpers'/script)],cwd=ROOT,text=True,capture_output=True)
  (OUT/f'{script}.log').write_text(res.stdout+'\n'+res.stderr);assert res.returncode==0,res.stderr[-3000:];print(script,round(time.time()-start,1),'seconds',flush=True)
 tests=[];ills=[];coeff=[];components=[]
 for name in ['september','june','shape_september','shape_june']:
  p=OUT/'ols'/name
  tab=pd.read_csv(p/'tests.csv');tab['analysis']=name;tests.append(tab)
  ill=pd.read_csv(p/'illustrations.csv');ill['analysis']=name;ills.append(ill)
  comp=pd.read_csv(p/'components.csv');comp['analysis']=name;components.append(comp)
  ref=ROOT/'provenance/reference_ols'/('' if name=='september' else name)/'tests.csv';old=pd.read_csv(ref)
  assert np.allclose(tab.composite_p,old.composite_p,rtol=1e-4,atol=1e-12)
  for key,f in json.loads((p/'fits.json').read_text()).items():
   f['beta']=np.asarray(f['beta']);f['covariance']=np.asarray(f['covariance'])
   for j,term in enumerate(f['terms']):
    coeff.append(dict(analysis=name,fit=key,term=term,n=f['n'],hosts=f['hosts'],**contrast(f,np.eye(3)[j])))
 tests=pd.concat(tests,ignore_index=True);ills=pd.concat(ills,ignore_index=True);coeff=pd.DataFrame(coeff);components=pd.concat(components,ignore_index=True)
 for df,name in [(tests,'ols_tests'),(ills,'ols_contrasts'),(coeff,'ols_focus_coefficients'),(components,'ols_components')]:df.to_csv(OUT/f'{name}.csv',index=False)
 assert len(tests)==40 and tests.query("analysis=='september' and spec=='host_fixed_effects' and split=='holdout'").supported.eq(False).all()
 # Contrast arithmetic and component combination independently checked from saved covariance.
 for _,g in components.groupby(['analysis','spec','split']):
  row=tests[(tests.analysis==g.analysis.iloc[0])&(tests.spec==g.spec.iloc[0])&(tests.split==g.split.iloc[0])].iloc[0]
  assert np.isclose(row.composite_p,g.p_two_sided.max()) and np.isclose(row.p_adjusted_21,min(1,21*g.p_two_sided.max()))
 q=ills.query("analysis=='september' and spec=='primary' and split=='holdout'")
 assert np.isclose(q.iloc[1].coefficient-q.iloc[0].coefficient,q.iloc[2].coefficient)
 fig,ax=plt.subplots(figsize=(9,4.5));labels=[]
 selected=[('september','primary','train','Сентябрь: train'),('september','primary','holdout','Сентябрь: старый test'),('june','primary','june','Июнь: все'),('june','new_hosts','june','Июнь: другие хозяева')]
 for j,(a,s,sp,label) in enumerate(selected):
  labels.append(label);q=ills[(ills.analysis==a)&(ills.spec==s)&(ills.split==sp)]
  for k,n in enumerate([10,100]):
   z=q[q.comparison==f'rating_plus_02_at_{n}_reviews'].iloc[0];pos=j+(-.12 if k==0 else .12);ax.errorbar(z['pct_change'],pos,xerr=[[z['pct_change']-z.pct_ci_low],[z.pct_ci_high-z['pct_change']]],fmt='o',color=COLORS[k],label=f'{n} отзывов' if j==0 else None,capsize=4)
 ax.set(yticks=range(4),yticklabels=labels,xlabel='Премия за рейтинг 4,7 → 4,9, %',title='OLS: обычные 95% интервалы с кластеризацией по хозяевам');ax.legend();fig.tight_layout();savefig('ols_contrasts')
 return tests,ills,coeff

def prediction(d,X,y,num,cat,ix):
 pre=preprocess(X,num,cat);mat={k:pre.fit_transform(X.iloc[i]) if k=='train' else pre.transform(X.iloc[i]) for k,i in ix.items()}
 candidates=[('OLS',{},LinearRegression(tol=1e-8))]+[('Ridge',{'alpha':a},Ridge(alpha=a,solver='lsqr',tol=1e-6)) for a in [1,10,100]]
 candidates += [('Small tree',{'max_depth':5,'min_samples_leaf':50},DecisionTreeRegressor(max_depth=5,min_samples_leaf=50,random_state=SEED)),('Random forest',{'n_estimators':200,'max_depth':14,'min_samples_leaf':10},RandomForestRegressor(n_estimators=200,max_depth=14,min_samples_leaf=10,n_jobs=4,random_state=SEED))]
 candidates += [('XGBoost',{'max_depth':dep,'n_estimators':300,'learning_rate':.05,'min_child_weight':20,'subsample':.8,'colsample_bytree':.8,'reg_lambda':10},xgb(dep)) for dep in [3,5]]
 selected={};rows=[];choices={}
 for name,pars,m in candidates:
  print('Prediction:',name,pars,flush=True);den=name in ['Small tree','Random forest'];m.fit(dense(mat['train']) if den else mat['train'],y[ix['train']]);pv=m.predict(dense(mat['validation']) if den else mat['validation']);sc=metrics(y[ix['validation']],pv)
  rows.append(dict(model=name,params=json.dumps(pars),**sc))
  if name not in selected or sc['log_mae']<selected[name][0]:selected[name]=(sc['log_mae'],m);choices[name]=pars
 pd.DataFrame(rows).to_csv(OUT/'tuning.csv',index=False);save_json('selected_settings.json',choices)
 predictions={};scores=[]
 for split,idx in ix.items():
  pp={'Median':np.full(len(idx),np.median(y[ix['train']]))}
  for name,(_,m) in selected.items():pp[name]=m.predict(dense(mat[split]) if name in ['Small tree','Random forest'] else mat[split])
  frame=d.iloc[idx][['id','host_id','room_type','number_of_reviews','review_scores_rating','price_gbp']].copy();frame['log_price']=y[idx]
  for name,pred in pp.items():frame[name]=pred;scores.append(dict(split=split,model=name,n=len(idx),**metrics(y[idx],pred)))
  frame.to_csv(OUT/f'predictions_{split}.csv',index=False);predictions[split]=frame
 scores=pd.DataFrame(scores);scores.to_csv(OUT/'model_scores.csv',index=False)
 joblib.dump(pre,ROOT/'models/preprocessor.joblib');selected['XGBoost'][1].save_model(ROOT/'models/xgboost.json')
 save_json('encoded_features.json',list(pre.get_feature_names_out()))
 old=pd.read_csv(ROOT/'provenance/reference_prediction/holdout_metrics.csv').set_index('model');new=scores[scores.split=='holdout'].set_index('model');mapping={'Median':'median','Ridge':'PL_ridge','Small tree':'PL_small_tree','Random forest':'PL_random_forest','XGBoost':'PL_xgboost'}
 rep=[]
 for n,o in mapping.items():
  for metric in ['log_mae','gbp_mae','log_r2']:
   delta=float(new.loc[n,metric]-old.loc[o,metric]);rep.append(dict(model=n,metric=metric,new=new.loc[n,metric],reference=old.loc[o,metric],difference=delta));assert abs(delta)<1e-5
 pd.DataFrame(rep).to_csv(OUT/'prediction_reproduction.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,4));new.loc[list(mapping),'log_mae'].sort_values().plot.barh(ax=ax,color=COLORS[0]);ax.set(xlabel='MAE логарифма цены, меньше — лучше',ylabel='',title='Одна выборка: 11 904 объявления старого test');fig.tight_layout();savefig('model_scores')
 return pre,selected,mat,scores,predictions

def bootstrap_errors(frame):
 rng=np.random.default_rng(SEED);codes,hosts=pd.factorize(frame.host_id);counts=np.bincount(codes);rows=[]
 for baseline in ['Ridge','Random forest']:
  a=np.abs(frame.log_price-frame[baseline]).to_numpy();b=np.abs(frame.log_price-frame.XGBoost).to_numpy();sums=np.bincount(codes,weights=a-b);boot=[]
  for _ in range(1000):
   k=rng.integers(len(hosts),size=len(hosts));boot.append(sums[k].sum()/counts[k].sum())
  rows.append(dict(comparison=f'XGBoost over {baseline}',log_mae_gain=(a-b).mean(),relative_pct=100*(a-b).mean()/a.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
 out=pd.DataFrame(rows);out.to_csv(OUT/'paired_host_bootstrap.csv',index=False);return out

def cv_experiment(d,X,y,num,cat,ix):
 dev=np.r_[ix['train'],ix['validation']];rows=[];membership=[]
 for fold,(a,b) in enumerate(GroupKFold(n_splits=3).split(dev,groups=d.iloc[dev].host_id),1):
  tr,va=dev[a],dev[b];assert not set(d.iloc[tr].host_id)&set(d.iloc[va].host_id)
  membership.extend([dict(id=int(i),fold=fold) for i in d.iloc[va].id]);pre=preprocess(X,num,cat);xt=pre.fit_transform(X.iloc[tr]);xv=pre.transform(X.iloc[va])
  for name,m in [('Ridge',Ridge(alpha=10,solver='lsqr',tol=1e-6)),('XGBoost',xgb())]:
   print('CV',fold,name,flush=True);m.fit(xt,y[tr]);rows.append(dict(fold=fold,model=name,n_train=len(tr),n_validation=len(va),**metrics(y[va],m.predict(xv))))
 out=pd.DataFrame(rows);out.to_csv(OUT/'development_cv.csv',index=False);pd.DataFrame(membership).to_csv(OUT/'cv_membership.csv',index=False);return out

def rated_design(d):
 use=d.room_type.isin(['Entire home/apt','Private room'])&d.review_scores_rating.between(3,5)&d.number_of_reviews.between(1,500);q=d[use].reset_index(drop=True);X,num,cat=prepare(q)
 for name,v in {'log_capacity':np.log(q.accommodates),'log_reviews':np.log1p(q.number_of_reviews.clip(lower=0)),'log_host_listings':np.log1p(q.host_listings_count.clip(lower=0)),'log_minimum_nights':np.log(q.minimum_nights),'distance5':X.distance_km/5,'beyond5':np.maximum(X.distance_km-5,0)/5}.items():X[name]=v
 num=[c for c in num if c not in ['accommodates','number_of_reviews','host_listings_count','minimum_nights']]+['log_capacity','log_reviews','log_host_listings','log_minimum_nights','distance5','beyond5','latitude','longitude'];cat+=['neighbourhood_cleansed']
 removed=['log_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication'];tr=np.flatnonzero(q.split.eq('train'));ns=[c for c in num if c not in removed and X.iloc[tr][c].notna().any() and X.iloc[tr][c].nunique()>1];cs=[c for c in cat if X.iloc[tr][c].nunique()>1]
 assert ns+cs==json.loads((OUT/'ols/september/controls.json').read_text())['R']
 X['rating_centered']=q.review_scores_rating-4.8;X['log_reviews_centered']=np.log1p(q.number_of_reviews)-np.log(11);X['rating_review_interaction']=X.rating_centered*X.log_reviews_centered
 return q,X,ns,cs

def ablation(d):
 q,X,ns,cs=rated_design(d);y=np.log(q.price_gbp.to_numpy());ix={s:np.flatnonzero(q.split.eq(s)) for s in ['train','validation','holdout']}
 base=[c for c in ns if c!='reviews_per_month'];main=ns+['rating_centered','log_reviews_centered'];full=main+['rating_review_interaction'];rows=[];preds=q[['id','host_id','split']].copy();models={}
 for label,nums in [('Non-review',base),('Review main',main),('Review interaction',full)]:
  pre=preprocess(X,nums,cs);xt=pre.fit_transform(X.iloc[ix['train']]);m=LinearRegression(tol=1e-10);m.fit(dense(xt),y[ix['train']]);pp=m.predict(dense(pre.transform(X)));preds['OLS_'+label]=pp
  for split,idx in ix.items():rows.append(dict(model='OLS',features=label,split=split,n=len(idx),**metrics(y[idx],pp[idx])))
  if label=='Review interaction':
   # Full primary-training coefficients and host-cluster covariance, using SVD minimum-norm solution.
   A=np.column_stack([np.ones(len(ix['train'])),dense(xt)]);yy=y[ix['train']];beta,_,rank,_=linalg.lstsq(A,yy,cond=1e-12,lapack_driver='gelsd');res=yy-A@beta;codes,hosts=pd.factorize(q.iloc[ix['train']].host_id);influence=linalg.pinv(A,rtol=1e-12).T;scores=np.zeros((len(hosts),A.shape[1]));np.add.at(scores,codes,influence*res[:,None]);cov=len(hosts)/(len(hosts)-1)*(len(yy)-1)/(len(yy)-rank)*(scores.T@scores);se=np.sqrt(np.maximum(0,np.diag(cov)));crit=stats.t.ppf(.975,len(hosts)-1)
   pd.DataFrame({'term':['intercept']+list(pre.get_feature_names_out()),'coefficient':beta,'cluster_se':se,'ci_low':beta-crit*se,'ci_high':beta+crit*se,'p_two_sided':2*stats.t.sf(np.abs(np.divide(beta,se,out=np.zeros_like(beta),where=se>0)),len(hosts)-1)}).to_csv(OUT/'ols_full_training_coefficients.csv',index=False)
   save_json('ols_full_design_note.json',{'n':len(yy),'columns':A.shape[1],'rank':int(rank),'hosts':len(hosts),'note':'Numeric columns standardized. Some encoded controls may be aliased; minimum-norm coefficients are descriptive. Focus inference uses unscaled residualized terms and verified covariance.'})
 for label,nums in [('Non-review',base),('Review information',main)]:
  pre=preprocess(X,nums,cs);xt=pre.fit_transform(X.iloc[ix['train']]);m=xgb();m.fit(xt,y[ix['train']]);pp=m.predict(pre.transform(X));preds['XGBoost_'+label]=pp
  for split,idx in ix.items():rows.append(dict(model='XGBoost',features=label,split=split,n=len(idx),**metrics(y[idx],pp[idx])))
  models[label]=(pre,m)
 out=pd.DataFrame(rows);out.to_csv(OUT/'matched_ablation.csv',index=False);preds.to_csv(OUT/'matched_predictions.csv',index=False)
 return out,q,X,models

def permutation(d,X,y,ix,pre,model):
 rng=np.random.default_rng(SEED);indices=rng.choice(ix['holdout'],size=2000,replace=False);sample=X.iloc[indices].copy();base=mean_absolute_error(y[indices],model.predict(pre.transform(sample)));groups={g:[c for c in X if feature_group(c)==g] for g in ['Location','Property','Host','Amenities','Reviews','Booking context']};rows=[]
 for g,cols in groups.items():
  cols=[c for c in cols if c in pre.feature_names_in_]
  for repeat in range(5):
   alt=sample.copy();order=rng.permutation(len(alt));alt.loc[:,cols]=sample.iloc[order][cols].to_numpy();err=mean_absolute_error(y[indices],model.predict(pre.transform(alt)));rows.append(dict(group=g,repeat=repeat,base_log_mae=base,permuted_log_mae=err,increase=err-base))
 out=pd.DataFrame(rows);out.to_csv(OUT/'permutation_repeats.csv',index=False);summary=out.groupby('group').increase.agg(['mean','std']).sort_values('mean',ascending=False);summary.to_csv(OUT/'permutation_summary.csv');d.iloc[indices][['id','host_id']].to_csv(OUT/'permutation_sample.csv',index=False)
 fig,ax=plt.subplots(figsize=(8,4));summary.sort_values('mean')['mean'].plot.barh(ax=ax,color=COLORS[0]);ax.set(xlabel='Рост log MAE после перемешивания группы',ylabel='',title='Групповая permutation importance');fig.tight_layout();savefig('permutation')
 return summary

def shap_analysis(d,X,y,ix,pre,model):
 rng=np.random.default_rng(SEED);ids=rng.choice(ix['holdout'],1000,replace=False);xx=pre.transform(X.iloc[ids]);contrib=model.get_booster().predict(DMatrix(xx),pred_contribs=True);prediction=model.predict(xx);assert np.allclose(contrib.sum(1),prediction,atol=2e-5)
 names=pre.get_feature_names_out();raw=list(pre.feature_names_in_);mapping=[]
 for name in names:
  suffix=name.split('__',1)[1].removeprefix('missingindicator_');c=next((c for c in sorted(raw,key=len,reverse=True) if suffix==c or suffix.startswith(c+'_')),None);assert c is not None,name;mapping.append(c)
 agg=pd.DataFrame({c:contrib[:,:-1][:,np.array(mapping)==c].sum(1) for c in dict.fromkeys(mapping)});assert np.allclose(agg.sum(1)+contrib[:,-1],prediction,atol=2e-5)
 agg.insert(0,'id',d.iloc[ids].id.to_numpy());agg.to_csv(OUT/'shap_contributions.csv',index=False);sample=d.iloc[ids][['id','host_id','room_type','price_gbp','review_scores_rating','number_of_reviews']].reset_index(drop=True);sample['prediction_log']=prediction;sample['base_log']=contrib[:,-1];sample['error_log']=prediction-y[ids];sample.to_csv(OUT/'shap_sample.csv',index=False)
 importance=agg.drop(columns='id').abs().mean().sort_values(ascending=False);importance.rename('mean_abs_shap').to_csv(OUT/'shap_importance.csv');top=importance.head(15).index[::-1]
 fig,ax=plt.subplots(figsize=(10,6))
 for j,c in enumerate(top):
  v=pd.to_numeric(X.iloc[ids][c],errors='coerce').to_numpy();finite=np.isfinite(v)
  if finite.any():lo,hi=np.nanquantile(v,[.05,.95]);v=np.clip((v-lo)/max(hi-lo,1e-9),0,1)
  else:v=np.full(len(v),.5)
  ax.scatter(agg[c],j+rng.uniform(-.28,.28,len(ids)),c=np.nan_to_num(v,nan=.5),cmap='coolwarm',vmin=0,vmax=1,s=6,alpha=.4)
 ax.axvline(0,color='#888888',lw=.7);ax.set(yticks=range(len(top)),yticklabels=top,xlabel='TreeSHAP: вклад в прогноз log(price)',title='1 000 случайных объявлений test; красный = больше значение');fig.tight_layout();savefig('shap_summary')
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,c,colorcol,label in [(axs[0],'review_scores_rating','number_of_reviews','Рейтинг'),(axs[1],'number_of_reviews','review_scores_rating','log(1 + отзывы)')]:
  x=sample[c].to_numpy();x=np.log1p(x) if c=='number_of_reviews' else x;col=sample[colorcol].to_numpy();col=np.log1p(col) if colorcol=='number_of_reviews' else col
  sc=ax.scatter(x,agg[c],c=col,cmap='viridis',s=10,alpha=.6);fig.colorbar(sc,ax=ax,label='log(1 + отзывы)' if colorcol=='number_of_reviews' else 'Рейтинг');ax.set(xlabel=label,ylabel='Вклад TreeSHAP, log(price)')
 fig.tight_layout();savefig('shap_dependence')
 errors=np.abs(sample.error_log);chosen=[];local=[];fig,axs=plt.subplots(1,3,figsize=(14,5))
 for ax,quantile in zip(axs,[.25,.5,.9]):
  idx=int(np.abs(errors-errors.quantile(quantile)).argmin());chosen.append(idx);s=agg.drop(columns='id').iloc[idx];terms=s.abs().nlargest(6).index;values=s[terms];other=s.drop(terms).sum();vals=pd.concat([values,pd.Series({'Other features':other})]);ax.barh(vals.index[::-1],vals.values[::-1],color=[COLORS[0] if v>=0 else COLORS[1] for v in vals.values[::-1]]);ax.axvline(0,color='gray',lw=.6);ax.set_title(f'Ошибка: квантиль {quantile:.0%}\nЦена £{sample.price_gbp.iloc[idx]:.0f}; прогноз £{np.exp(prediction[idx]):.0f}');ax.set_xlabel('Вклад в log(price)')
  local.append(dict(sample.iloc[idx],selection_error_quantile=quantile,contribution_sum=float(s.sum()),top_features='; '.join(terms)))
 fig.tight_layout();savefig('local_explanations');pd.DataFrame(local).to_csv(OUT/'local_examples.csv',index=False)
 save_json('shap_method.json',{'method':'Exact native XGBoost TreeSHAP, pred_contribs=True, approx_contribs=False','background':'Training tree path frequencies (tree-path-dependent), no separate interventional background','sample':'Uniform 1000 rows without replacement from original holdout, seed20261016','units':'log(price)','aggregation':'one-hot and missingness indicators summed to source feature; additivity verified','max_additivity_error':float(np.abs(contrib.sum(1)-prediction).max())})
 return importance,sample,agg

def response(q,X,models):
 pre,model=models['Review information'];rng=np.random.default_rng(SEED);test=q.split.eq('holdout');support=[]
 for n,lo,hi in [(10,5,20),(100,70,130)]:
  for rating in [4.7,4.9]:support.append(dict(reviews=n,rating=rating,n=int((test&q.number_of_reviews.between(lo,hi)&q.review_scores_rating.between(rating-.05,rating+.05)).sum()),review_low=lo,review_high=hi))
 support=pd.DataFrame(support);support.to_csv(OUT/'response_support.csv',index=False);assert support.n.ge(30).all()
 eligible=np.flatnonzero(test&q.review_scores_rating.between(4.5,5)&q.number_of_reviews.between(5,150));ids=rng.choice(eligible,min(1000,len(eligible)),replace=False);base=X.iloc[ids].copy();individual=q.iloc[ids][['id','host_id']].copy();rows=[]
 for n in [10,100]:
  predictions=[]
  for rating in [4.7,4.9]:
   alt=base.copy();alt['rating_centered']=rating-4.8;alt['log_reviews_centered']=np.log1p(n)-np.log(11);predictions.append(model.predict(pre.transform(alt)))
  delta=predictions[1]-predictions[0];individual[f'delta_log_{n}']=delta;rows.append(dict(reviews=n,mean_delta_log=float(delta.mean()),premium_pct=float(100*np.expm1(delta.mean())),ice_q10_pct=float(np.quantile(100*np.expm1(delta),.1)),ice_q90_pct=float(np.quantile(100*np.expm1(delta),.9)),n=len(ids)))
 individual.to_csv(OUT/'response_individual.csv',index=False);out=pd.DataFrame(rows);out.to_csv(OUT/'response_contrasts.csv',index=False)
 grid=[]
 for rating in np.linspace(4.5,5,6):
  for n in [5,10,20,50,100,150]:
   count=int((test&q.review_scores_rating.between(rating-.05,rating+.05)&q.number_of_reviews.between(max(1,n*.7),n*1.3)).sum());val=np.nan
   if count>=30:
    alt=base.copy();alt['rating_centered']=rating-4.8;alt['log_reviews_centered']=np.log1p(n)-np.log(11);val=float(model.predict(pre.transform(alt)).mean())
   grid.append(dict(rating=rating,reviews=n,observed_support=count,mean_predicted_log_price=val))
 grid=pd.DataFrame(grid);grid.to_csv(OUT/'response_grid.csv',index=False)
 fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].bar(out.reviews.astype(str),out.premium_pct,color=COLORS[:2]);axs[0].set(xlabel='Число отзывов',ylabel='Изменение модельного прогноза, %',title='XGBoost: фиксированный рейтинг 4,7 → 4,9');pivot=grid.pivot(index='rating',columns='reviews',values='mean_predicted_log_price');im=axs[1].imshow(pivot,aspect='auto',origin='lower',cmap='viridis');axs[1].set(xticks=range(6),xticklabels=pivot.columns,yticks=range(6),yticklabels=[f'{v:.1f}' for v in pivot.index],xlabel='Число отзывов',ylabel='Рейтинг',title='Средний log-прогноз; белые клетки: мало данных');fig.colorbar(im,ax=axs[1]);fig.tight_layout();savefig('response')
 return out,support

def error_audit(frame):
 e=frame.copy();e['absolute_log_error']=np.abs(e.XGBoost-e.log_price);e['absolute_gbp_error']=np.abs(np.exp(e.XGBoost)-e.price_gbp);e['price_band']=pd.cut(e.price_gbp,[0,100,200,400,2000],labels=['≤100','100–200','200–400','400–2000']);e['review_band']=pd.cut(e.number_of_reviews,[-1,0,10,100,100000],labels=['0','1–10','11–100','101+']);rows=[]
 for c in ['room_type','price_band','review_band']:
  g=e.groupby(c,observed=True).agg(n=('id','size'),log_mae=('absolute_log_error','mean'),gbp_mae=('absolute_gbp_error','mean')).reset_index().rename(columns={c:'group'});g.insert(0,'dimension',c);rows.append(g)
 out=pd.concat(rows);out.to_csv(OUT/'error_audit.csv',index=False);return out
