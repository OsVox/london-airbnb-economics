"""Plot the two SHAP components and test one explicitly added OLS interaction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


def bathroom_shap_comparison(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    p=pd.read_csv(root/'results/bathroom_shap_pair_sample.csv')
    p=p[p.room_type.isin(['Entire home/apt','Private room'])&p.bathroom_count.isin([1,1.5,2])].copy()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    f,ax=plt.subplots(figsize=(11,4.6));rng=np.random.default_rng(20261018)
    jitter=rng.uniform(-.12,.12,len(p));norm=Normalize(vmin=1,vmax=2);cmap=plt.cm.coolwarm
    for y,feature in [(1,'bathroom_shap'),(0,'full_pair_shap')]:
        for room,marker in [('Entire home/apt','o'),('Private room','^')]:
            mask=p.room_type.eq(room).to_numpy()
            ax.scatter(p.loc[mask,feature],y+jitter[mask],c=p.loc[mask,'bathroom_count'],
                       cmap=cmap,norm=norm,marker=marker,s=20,alpha=.55,edgecolors='none')
    ax.axvline(0,color='#777',lw=1)
    ax.set(yticks=[1,0],yticklabels=['Весь SHAP-вклад\nчисла ванных','SHAP-взаимодействие\nс типом жилья'],
           ylim=(-.43,1.38),xlim=(-.20,.165),xlabel='Вклад в прогноз log(цены)')
    ax.spines[['top','right','left']].set_visible(False)
    ax.set_title('Красные точки — ровно две ванные',fontsize=16,pad=18)
    bar=f.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,pad=.025,fraction=.025,ticks=[1,1.5,2])
    bar.ax.set_yticklabels(['1','1,5','2']);bar.set_label('Число ванных')
    legend=[Line2D([0],[0],marker='o',linestyle='',color='#333',label='Целая квартира',markersize=6),
            Line2D([0],[0],marker='^',linestyle='',color='#333',label='Отдельная комната',markersize=7)]
    ax.legend(handles=legend,loc='upper left',fontsize=9)
    f.text(.5,.025,f'Те же {len(p)} объявлений в обеих строках. Сверху: красные точки только справа от нуля.\n'
           'Снизу: парный вклад двух ванных положителен для квартиры и отрицателен для комнаты.',ha='center',fontsize=10)
    f.tight_layout(rect=(0,.12,1,1));f.savefig(root/'figures/bathroom_shap_comparison.png',dpi=165,bbox_inches='tight');plt.close(f)


def linear_interaction_check(d,X,pre,ols,xgb,root=None):
    """One exploratory specification change; only the original train is fitted."""
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    train=d.split.eq('train').to_numpy()
    count=X.bathroom_count.fillna(X.loc[train,'bathroom_count'].median())
    term=(count*X.room_type.eq('Private room')).to_numpy()[:,None]
    matrix=pre.transform(X)
    extended=sparse.hstack([matrix,sparse.csr_matrix(term)],format='csr')
    fitted=LinearRegression(tol=1e-8)
    fitted.fit(extended[train],np.log(d.loc[train,'price_gbp']))
    predictions={'OLS':ols.predict(matrix),'OLS + bathroom_count x private room':fitted.predict(extended),
                 'XGBoost':xgb.predict(matrix)}
    rows=[]
    for split in ['validation','holdout']:
        mask=d.split.eq(split).to_numpy()
        for name,values in predictions.items():
            rows.append({'split':split,'model':name,'n':int(mask.sum()),
                         'log_mae':mean_absolute_error(np.log(d.loc[mask,'price_gbp']),values[mask]),
                         'gbp_mae':mean_absolute_error(d.loc[mask,'price_gbp'],np.exp(values[mask]))})
    result=pd.DataFrame(rows)
    result.to_csv(root/'results/bathroom_ols_interaction_check.csv',index=False)
    metadata={'status':'Exploratory specification check after inspecting existing test results',
              'term':'bathroom_count * I(room_type == Private room)',
              'training':'Original train only, original fitted preprocessing, no tuning',
              'missing_bathroom_count':'Training median used for the product term',
              'new_term_coefficient':float(fitted.coef_[-1]),
              'interpretation':'Correction of an omitted interaction, not proof of absence of unobserved confounding'}
    (root/'results/bathroom_ols_interaction_method.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    return result
