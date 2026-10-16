from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from ols_core import estimate,contrast
H=Path(__file__).resolve().parents[1];R=H;O=H/'results/ols/september'
O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'helpers'))
from features import prepare
paths=[H/'provenance/analysis_plan.md',Path(__file__),H/'helpers/ols_core.py',R/'helpers/features.py',R/'data/london_2026-09-17.csv.gz',R/'data/split.csv']
(O/'provenance.json').write_text(json.dumps({str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
d=pd.read_csv(R/'data/london_2026-09-17.csv.gz',low_memory=False).merge(pd.read_csv(R/'data/split.csv'),on=['id','host_id'],validate='one_to_one')
X,_,_=prepare(d)
use=d.room_type.isin(['Entire home/apt','Private room'])&d.review_scores_rating.between(3,5)&d.number_of_reviews.between(1,500)
d=d[use].reset_index(drop=True);X,num,cat=prepare(d);price=pd.to_numeric(d.price.astype(str).str.replace(r'[^0-9.\-]','',regex=True)).to_numpy();y=np.log(price)
assert d.id.is_unique and np.all((price>=10)&(price<=2000))
for a,b in [('train','holdout'),('train','validation'),('validation','holdout')]:assert not set(d.loc[d.split.eq(a),'host_id'])&set(d.loc[d.split.eq(b),'host_id'])
xkm=6371*np.cos(np.radians(51.5074))*np.radians(d.longitude+.1278);ykm=6371*np.radians(d.latitude-51.5074);d['grid_1km']=np.floor(xkm).astype(int).astype(str)+'_'+np.floor(ykm).astype(int).astype(str)
d[['id','host_id','split','grid_1km']].to_csv(O/'membership.csv',index=False)
for name,v in {'log_capacity':np.log(d.accommodates),'log_reviews':np.log1p(d.number_of_reviews.clip(lower=0)),'log_host_listings':np.log1p(d.host_listings_count.clip(lower=0)),'log_minimum_nights':np.log(d.minimum_nights),'distance5':X.distance_km/5,'beyond5':np.maximum(X.distance_km-5,0)/5}.items():X[name]=v
num=[c for c in num if c not in ['accommodates','number_of_reviews','host_listings_count','minimum_nights']]+['log_capacity','log_reviews','log_host_listings','log_minimum_nights','distance5','beyond5','latitude','longitude'];cat+=['neighbourhood_cleansed']
review_gap=np.log(101)-np.log(11)
models={'R':{'T':np.column_stack([d.review_scores_rating-4.8,np.log1p(d.number_of_reviews)-np.log(11),(d.review_scores_rating-4.8)*(np.log1p(d.number_of_reviews)-np.log(11))]),'remove':['log_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication'],'terms':['rating_centered','log_reviews_centered','rating_review_interaction'],'components':[('rating_slope_at_100_reviews',[1,0,review_gap],0,1),('rating_slope_100_minus_10',[0,0,review_gap],0,1)],'illustrations':[('rating_plus_02_at_10_reviews',[.2,0,0],0),('rating_plus_02_at_100_reviews',[.2,0,.2*review_gap],0),('rating_premium_ratio_100_vs_10',[0,0,.2*review_gap],0)]}}
rows=[];components=[];ill=[];fits={};controls={}
train=np.flatnonzero(d.split.eq('train'))
for h,m in models.items():
 ns=[c for c in num if c not in m['remove'] and X.iloc[train][c].notna().any() and X.iloc[train][c].nunique()>1];cs=[c for c in cat if X.iloc[train][c].nunique()>1]
 pre=ColumnTransformer([('num',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()),ns),('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=20,drop='first'),cs)],sparse_threshold=1.)
 pre.fit(X.iloc[train]);mat=pre.transform(X);mat=mat.toarray() if sparse.issparse(mat) else mat;W=np.column_stack([np.ones(len(d)),mat]);names=['intercept']+list(pre.get_feature_names_out());T=m['T'];controls[h]=ns+cs
 curve=[j for j,c in enumerate(names) if c.startswith('num__') and not c.startswith('num__missingindicator_') and not c.startswith('num__amenity_') and X.iloc[train][c.removeprefix('num__')].nunique()>2];Q=np.column_stack([W,W[:,curve]**2])
 subcols=['review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication']; subpre=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler()); subpre.fit(X.iloc[train][subcols]); W_sub=np.column_stack([W,subpre.transform(X[subcols])])
 cases=[('primary',s) for s in ['train','validation','holdout']]+[(spec,s) for spec in ['quadratic_controls','equal_host_weights','borough_cluster','middle_price_range','host_fixed_effects','grid_fixed_effects','rating_subscores','entire_home_only','private_room_only'] for s in ['train','holdout']]
 for spec,split in cases:
  use=d.split.eq(split)
  if spec=='entire_home_only':use=use&d.room_type.eq('Entire home/apt')
  if spec=='private_room_only':use=use&d.room_type.eq('Private room')
  if spec=='middle_price_range':use=use&(price>=20)&(price<=1000)
  index=np.flatnonzero(use);weights=None;matrix=Q if spec=='quadratic_controls' else W_sub if spec=='rating_subscores' else W
  if spec=='equal_host_weights':
   weights=1/d.iloc[index].groupby('host_id').id.transform('size').to_numpy();weights/=weights.mean()
  cluster='neighbourhood_cleansed' if spec=='borough_cluster' else 'host_id'
  if spec in ['host_fixed_effects','grid_fixed_effects']:
   absorb='host_id' if spec=='host_fixed_effects' else 'grid_1km';dd=d.iloc[index].copy();dd=dd[dd.groupby(absorb).id.transform('size').ge(2)];index=dd.index.to_numpy();dd=dd.reset_index(drop=True);codes,labels=pd.factorize(dd[absorb]);g=len(labels);counts=np.bincount(codes)
   def demean(a):
    sums=np.zeros((g,a.shape[1]));np.add.at(sums,codes,a);return a-sums[codes]/counts[codes,None]
   ww=demean(W[index]);tt=demean(T[index]);yy=demean(y[index,None]).ravel();ww=ww[:,np.linalg.norm(ww,axis=0)>1e-9]
   f=estimate(ww,tt,np.arange(len(dd)),dd,yy,direct_check=True);oldrank=f['rank'];assert f['n']>oldrank+g;f['covariance']*=(f['n']-oldrank)/(f['n']-oldrank-g);f['rank']+=g;f['absorbed_effects']=g
  else:f=estimate(matrix,T,index,d,y,weights,cluster,direct_check=(spec=='primary' and split=='train'))
  fits[h+'__'+spec+'__'+split]={k:v.tolist() if isinstance(v,np.ndarray) else int(v) if isinstance(v,np.integer) else v for k,v in f.items()};fits[h+'__'+spec+'__'+split]['terms']=m['terms']
  parts=[]
  for name,c,null,direction in m['components']:
   z=contrast(f,c,null);z.update(hypothesis=h,spec=spec,split=split,component=name,contrast=json.dumps(np.asarray(c,dtype=float).tolist()),direction=direction,expected_direction=bool(direction*(z['coefficient']-null)>0),ci_expected=bool(z['ci_low']>null if direction>0 else z['ci_high']<null));parts.append(z);components.append(z)
  for name,c,null in m['illustrations']:
   z=contrast(f,c,null);z.update(hypothesis=h,spec=spec,split=split,comparison=name,contrast=json.dumps(np.asarray(c,dtype=float).tolist()));ill.append(z)
  rows.append(dict(hypothesis=h,spec=spec,split=split,n=f['n'],hosts=f['hosts'],clusters=f['clusters'],composite_p=max(z['p_two_sided'] for z in parts),all_directions=all(z['expected_direction'] for z in parts),all_cis_expected=all(z['ci_expected'] for z in parts)))
  print(h,spec,split,[(z['component'],round(z['coefficient'],4),z['p_two_sided']) for z in parts],flush=True)
table=pd.DataFrame(rows);table['p_adjusted_21']=np.minimum(1,21*table.composite_p);table['supported']=table.all_directions&table.p_adjusted_21.lt(.05)
table.to_csv(O/'tests.csv',index=False);pd.DataFrame(components).to_csv(O/'components.csv',index=False);pd.DataFrame(ill).to_csv(O/'illustrations.csv',index=False);(O/'fits.json').write_text(json.dumps(fits,indent=2));(O/'controls.json').write_text(json.dumps(controls,indent=2))
(O/'checks.json').write_text(json.dumps({'host_splits_disjoint':True,'primary_direct_design_checks':True,'fixed_effects_direct_design_checks':True,'all_orthogonality_and_rank_checks':True,'price_rating_review_cohort_bounds':True},indent=2))
print(table[table.spec.eq('primary')].to_string(index=False),flush=True)
