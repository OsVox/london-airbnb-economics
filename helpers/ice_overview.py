"""Exploratory centered ICE for numeric/amenity inputs of the fitted main models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Fixed before this overview; not tuned to obtain more sign reversals.
SPECS = [
 ('accommodates','Вместимость',[2,3,4,6]),
 ('bedrooms','Спальни',[1,2]),
 ('bathroom_count','Число ванных',[1,1.5,2]),
 ('reviews_per_month','Отзывов в месяц',[.5,1,2,3,5]),
 ('review_scores_cleanliness','Оценка чистоты',[4,4.5,4.8,5]),
 ('review_scores_rating','Общий рейтинг',[4.5,4.7,4.9,5]),
 ('host_days','Стаж хозяина, лет',[180,365,730,1825]),
 ('host_listings_count','Объявлений у хозяина',[1,2,5,10,20]),
 ('amenity_dishwasher','Посудомоечная машина',[0,1]),
 ('amenity_air_conditioning','Кондиционер',[0,1]),
 ('amenity_dedicated_workspace','Рабочее место',[0,1]),
 ('amenity_dryer','Сушильная машина',[0,1]),
]
ROOMS = ['Entire home/apt','Private room']
COLORS = ['#176B82','#D48743']


def ice_overview(d,X,pre,xgb,ols,root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    out,fig=root/'results',root/'figures'
    sample=pd.read_csv(out/'shap_sample.csv')
    pos=pd.Index(d.id).get_indexer(sample.id)
    assert (pos>=0).all() and d.iloc[pos].split.eq('holdout').all()
    sx=X.iloc[pos].reset_index(drop=True)
    rng=np.random.default_rng(20261020)
    long,summary,support=[],[],[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    f,axes=plt.subplots(4,3,figsize=(15,13))
    for ax,(feature,label,grid) in zip(axes.flat,SPECS):
        values=pd.to_numeric(sx[feature],errors='coerce')
        eligible=sx.room_type.isin(ROOMS)&values.between(grid[0],grid[-1])
        if feature=='accommodates':
            eligible &= sx.bedrooms.le(2)
        if feature=='bedrooms':
            eligible &= sx.accommodates.between(2,6)
        q=sx.loc[eligible].copy();ids=sample.loc[eligible,'id'].to_numpy()
        predictions=[];linear=[]
        for value in grid:
            alt=q.copy();alt[feature]=float(value)
            assert alt.drop(columns=feature).equals(q.drop(columns=feature))
            transformed=pre.transform(alt)
            predictions.append(xgb.predict(transformed))
            linear.append(ols.predict(transformed))
        predictions=np.stack(predictions,axis=1);linear=np.stack(linear,axis=1)
        delta=predictions-predictions[:,[0]];linear_delta=linear-linear[:,[0]]
        assert np.allclose(delta[:,0],0) and np.isfinite(delta).all()
        percent=100*np.expm1(delta)
        endpoint=delta[:,-1];threshold=np.log1p(.005)
        plus=(endpoint>threshold);minus=(endpoint<np.log1p(-.005))
        for group,mask in [('All',np.ones(len(q),dtype=bool))]+[(r,q.room_type.eq(r).to_numpy()) for r in ROOMS]:
            if not mask.any():continue
            summary.append({'feature':feature,'label':label,'group':group,'low':grid[0],'high':grid[-1],
                'n':int(mask.sum()),'xgb_change_pct':float(100*np.expm1(endpoint[mask].mean())),
                'ols_change_pct':float(100*np.expm1(linear_delta[mask,-1].mean())),
                'positive_share':float(plus[mask].mean()),'negative_share':float(minus[mask].mean()),
                'near_zero_share':float((~(plus|minus))[mask].mean())})
        for j,value in enumerate(grid):
            long.append(pd.DataFrame({'feature':feature,'id':ids,'room_type':q.room_type.to_numpy(),
                'original_value':q[feature].to_numpy(),'grid_value':value,'baseline_value':grid[0],
                'prediction_log':predictions[:,j],'delta_log':delta[:,j],
                'change_pct':percent[:,j],'ols_delta_log':linear_delta[:,j]}))
            # Counts are coarse marginal support, not proof of joint support.
            train=d.split.eq('train')&X.room_type.isin(ROOMS)
            observed=pd.to_numeric(X[feature],errors='coerce')
            width=0 if feature in ['accommodates','bedrooms','bathroom_count'] or feature.startswith('amenity_') else (grid[-1]-grid[0])*.1
            for room in ROOMS:
                count=int((train&X.room_type.eq(room)&observed.between(value-width,value+width)).sum())
                support.append({'feature':feature,'grid_value':value,'room_type':room,'window_halfwidth':width,'train_rows':count})
        if feature=='host_days':plotgrid=np.array(grid)/365.25
        else:plotgrid=np.array(grid)
        selected=rng.choice(len(q),size=min(120,len(q)),replace=False)
        for i in selected:ax.plot(plotgrid,percent[i],color='#6F7B84',alpha=.13,lw=.7)
        ax.plot(plotgrid,100*np.expm1(delta.mean(0)),color='#176B82',lw=2.3)
        ax.plot(plotgrid,100*np.expm1(linear_delta.mean(0)),color='#D48743',lw=1.8,ls='--')
        ax.axhline(0,color='#9A9A9A',lw=.7)
        ax.set_title(f'{label} · n={len(q)}',loc='left',fontsize=11)
        ax.set_xticks(plotgrid)
        if feature.startswith('amenity_'):ax.set_xticklabels(['Нет','Есть'])
        elif feature=='host_days':ax.set_xticklabels([f'{v:.1f}' for v in plotgrid])
        ax.set_ylabel('Изменение прогноза, %',fontsize=9)
        ax.text(.98,.03,f'↑ {plus.mean():.0%}   ↓ {minus.mean():.0%}',transform=ax.transAxes,
                ha='right',va='bottom',fontsize=9,bbox={'facecolor':'white','alpha':.8,'edgecolor':'none'})
        ax.spines[['top','right']].set_visible(False)
        if feature=='bathroom_count':
            ax.set_facecolor('#FFF6E8');ax.title.set_color('#A85514')
    legend=[Line2D([0],[0],color='#6F7B84',alpha=.5,lw=1,label='XGBoost: отдельные объявления'),
            Line2D([0],[0],color='#176B82',lw=2.3,label='XGBoost: сводная линия'),
            Line2D([0],[0],color='#D48743',lw=1.8,ls='--',label='OLS: общий ответ')]
    f.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,frameon=False)
    f.suptitle('ICE: как прогноз отвечает на изменение разных признаков',fontsize=17,y=.995)
    f.text(.5,.012,'Каждая кривая начинается с нуля при первом значении по горизонтали. Остальные входы фиксированы.\n'
           '↑ / ↓ — доли ответов выше +0,5% / ниже −0,5% между крайними значениями. Шкалы Y различаются.\n'
           'Сводные линии: средняя Δlog-прогноза, переведённая в проценты. Это не причинные эффекты.',ha='center',fontsize=10)
    f.tight_layout(rect=(0,.07,1,.94));f.savefig(fig/'ice_overview.png',dpi=150,bbox_inches='tight');plt.close(f)
    long=pd.concat(long,ignore_index=True);summary=pd.DataFrame(summary);support=pd.DataFrame(support)
    long.to_csv(out/'ice_overview_curves.csv',index=False);summary.to_csv(out/'ice_overview_summary.csv',index=False);support.to_csv(out/'ice_overview_support.csv',index=False)
    (out/'ice_overview_method.json').write_text(json.dumps({
        'status':'Exploratory overview of already examined test data; bathroom pattern already known',
        'sample':'Original uniform 1000 SHAP test rows, then feature-specific eligibility',
        'eligibility':'Entire homes/private rooms, observed feature inside the displayed range; capacity/bedroom restrictions in code',
        'grid':{feature:grid for feature,_,grid in SPECS},'seed':20261020,
        'centering':'f(z,x_other)-f(grid_min,x_other); percent=100*expm1(delta)',
        'summary':'100*expm1(mean delta log), not arithmetic mean of percentage responses',
        'sign_threshold':'positive: delta_log > log(1.005); negative: delta_log < log(0.995)',
        'support':'Marginal training counts in recorded windows, not joint support or a causal validity check',
        'cautions':'Correlated inputs may create unusual combinations; group populations/ranges differ, so not a feature-importance ranking',
        'training':'No refitting or model selection',
        'exclusions':'No coordinate/distance or quote calendar changes, because dependent derived columns would require coordinated changes'
    },ensure_ascii=False,indent=2)+'\n')
    return long,summary


def bathroom_ice_focus(root=None):
    """Two-point cICE using the freshly computed full-test bathroom contrasts."""
    root=Path(root) if root is not None else Path(__file__).resolve().parents[1]
    data=pd.read_csv(root/'results/bathroom_case_individual.csv')
    data=data[data.split.eq('holdout')]
    f,ax=plt.subplots(figsize=(9,4.8));rng=np.random.default_rng(20261020)
    for room,label,color in zip(ROOMS,['Целая квартира','Отдельная комната'],COLORS):
        q=data[data.room_type.eq(room)]
        selected=q.iloc[rng.choice(len(q),size=min(100,len(q)),replace=False)]
        for delta in selected.xgb_delta_log:
            ax.plot([1,2],[0,100*np.expm1(delta)],color=color,alpha=.10,lw=.8)
        mean=100*np.expm1(q.xgb_delta_log.mean())
        ax.plot([1,2],[0,mean],lw=3,color=color,marker='o',label=f'{label} · n={len(q)}')
        ax.annotate(f'{mean:+.2f}%',(2,mean),xytext=(10,0),textcoords='offset points',va='center',color=color,fontsize=12)
    ax.axhline(0,color='#777777',lw=.8)
    ax.set(xticks=[1,2],xticklabels=['1 ванная','2 ванные'],xlim=(.95,2.18),
           ylabel='Изменение прогноза, %\nотносительно одной ванной',
           title='Аномалия ICE: тип жилья меняет направление ответа')
    ax.legend(loc='upper left',fontsize=10);ax.spines[['top','right']].set_visible(False)
    f.text(.5,.01,'Тонкие линии — по 100 test-объектов каждого типа; сводные линии рассчитаны по всем объектам группы.\n'
           'Ни одна другая характеристика внутри сравнения не меняется. Это поведение модели, не эффект ремонта.',ha='center',fontsize=9)
    f.tight_layout(rect=(0,.10,1,1));f.savefig(root/'figures/ice_bathroom_focus.png',dpi=160,bbox_inches='tight');plt.close(f)
