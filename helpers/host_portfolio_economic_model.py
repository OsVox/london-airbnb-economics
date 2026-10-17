"""Earlier different-elasticity illustration; current model is host_portfolio_cost_model.py.

This alternative is retained for reproducibility. Its parameters were not estimated from Airbnb.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    'p0': 40., 'B': 40., 'tau_days': 30.,
    'small_N': 1, 'large_N': 20,
    'small_cost': 50., 'large_cost': 35.,
    'small_elasticity': 3., 'large_elasticity': 2.,
}
SMALL, LARGE, TOTAL, COST = '#176B82', '#9866A4', '#176B82', '#D48743'


def cost(N, p=PARAMETERS):
    """Illustrative economies of scale; intended domain N in [1,20]."""
    k = (p['small_cost']-p['large_cost']) / np.log(p['large_N']/p['small_N'])
    return p['small_cost'] - k*np.log(np.asarray(N)/p['small_N'])


def waiting_value(L, p=PARAMETERS):
    """Common expected NET future payoff, including future costs already."""
    return p['B']*(-np.expm1(-np.asarray(L)/p['tau_days']))


def booking_probability(price, epsilon, p=PARAMETERS):
    price = np.asarray(price)
    return np.minimum(1., (p['p0']/price)**epsilon)


def optimal_price(c, epsilon, L, p=PARAMETERS):
    mu = epsilon/(epsilon-1)
    return np.maximum(p['p0'], mu*(c+waiting_value(L,p)))


def expected_profit(price, c, epsilon, L, p=PARAMETERS):
    q = booking_probability(price,epsilon,p)
    return q*(price-c)+(1-q)*waiting_value(L,p)


def build(root=ROOT):
    root = Path(root); out, fig = root/'results', root/'figures'
    out.mkdir(exist_ok=True); fig.mkdir(exist_ok=True)
    p = PARAMETERS
    cs, cb, es, eb = p['small_cost'], p['large_cost'], p['small_elasticity'], p['large_elasticity']
    mus, mub = es/(es-1), eb/(eb-1)
    assert np.isclose(cost(1),cs) and np.isclose(cost(20),cb)
    assert 0<cb<cs and 1<eb<es
    Vstar = (mus*cs-mub*cb)/(mub-mus)
    assert 0<Vstar<p['B']
    Lstar = -p['tau_days']*np.log1p(-Vstar/p['B'])
    numerical_cross = brentq(lambda L: optimal_price(cb,eb,L)-optimal_price(cs,es,L),0,365)
    assert np.isclose(Lstar,numerical_cross)

    days=np.unique(np.r_[np.linspace(0,90,361),[0,1,7,14,30,60,90,Lstar]])
    V=waiting_value(days)
    ps=optimal_price(cs,es,days);pb=optimal_price(cb,eb,days)
    # Exact algebraic accounting at the small host's markup factor. This is not identification.
    cost_part=mus*(cb-cs)*np.ones_like(days)
    markup_part=(mub-mus)*(cb+V)
    assert np.allclose(pb-ps,cost_part+markup_part)
    assert np.all(ps>p['p0']) and np.all(pb>p['p0'])
    data=pd.DataFrame({'lead_days':days,'common_V_gbp':V,'small_price_gbp':ps,
        'large_price_gbp':pb,'gap_gbp':pb-ps,'gap_pct':100*(pb/ps-1),
        'cost_component_gbp':cost_part,'markup_component_gbp':markup_part})
    data.to_csv(out/'host_portfolio_model_prices.csv',index=False)
    example_days=[0,7,14,30,60]
    examples=data[data.lead_days.isin(example_days)].copy()
    examples.to_csv(out/'host_portfolio_model_examples.csv',index=False)

    checks=[]
    for c,eps in [(cs,es),(cb,eb)]:
        for L in [0,1,7,Lstar,14,30,60,90,365]:
            closed=float(optimal_price(c,eps,L))
            numerical=minimize_scalar(lambda price:-expected_profit(price,c,eps,L),
                bounds=(p['p0'],1000.),method='bounded',options={'xatol':1e-9}).x
            assert np.isclose(closed,numerical,atol=1e-4)
            probability=float(booking_probability(closed,eps))
            assert 0<probability<1
            mu=eps/(eps-1)
            # derivative of pi: q(p)/p * [(1-eps)*p + eps*(c+V)].
            left=(1-eps)*(closed*.99)+eps*(c+waiting_value(L))
            right=(1-eps)*(closed*1.01)+eps*(c+waiting_value(L))
            assert left>0 and right<0
            checks.append(dict(cost=c,epsilon=eps,lead_days=float(L),closed_price=closed,
                               numeric_price=float(numerical),booking_probability=probability))
    pd.DataFrame(checks).to_csv(out/'host_portfolio_model_checks.csv',index=False)

    scenarios=[]
    for label,csmall,cbig,epssmall,epsbig in [
        ('Both mechanisms',cs,cb,es,eb),
        ('Scale economy only',cs,cb,es,es),
        ('Different demand only',cs,cs,es,eb),
        ('Neither mechanism',cs,cs,es,es)]:
        a=optimal_price(csmall,epssmall,days);b=optimal_price(cbig,epsbig,days)
        scenarios.append(pd.DataFrame({'scenario':label,'lead_days':days,'gap_gbp':b-a}))
    scenarios=pd.concat(scenarios,ignore_index=True)
    scenarios.to_csv(out/'host_portfolio_model_scenarios.csv',index=False)
    scale=scenarios[scenarios.scenario.eq('Scale economy only')]
    demand=scenarios[scenarios.scenario.eq('Different demand only')]
    assert (scale.gap_gbp<0).all() and (demand.gap_gbp>0).all()
    assert data.iloc[0].gap_gbp<0 and data.iloc[-1].gap_gbp>0
    method={
        'status':'Proposed theoretical illustration; not estimated, calibrated, or validated against Airbnb',
        'parameters':p,'scale_cost_k':float((cs-cb)/np.log(20)),
        'types':'Two discrete host types: N=1 and N=20; no elasticity function for intermediate portfolios is assumed',
        'waiting_value':'V(L)=B*(1-exp(-L/tau)); same expected NET continuation payoff for both types',
        'waiting_microfoundation':'Optional common future net offer B, arrival intensity 1/tau; no common gross future price is imposed',
        'scope':'One current potential guest, one available night, risk neutrality; later pricing is summarized by an exogenous continuation value',
        'probability':'q(p)=(p0/p)^epsilon for p>=p0; q=1 below p0',
        'objective':'q(p)*(p-c)+(1-q(p))*V(L)',
        'global_solution':'max(p0,epsilon/(epsilon-1)*(c+V(L))); all illustrated optima are interior',
        'small_markup_factor':mus,'large_markup_factor':mub,
        'waiting_threshold_gbp':float(Vstar),'lead_threshold_days':float(Lstar),
        'gap_function_gbp':'15 - 20*exp(-L/30)',
        'gap_limit_gbp':float(mub*cb-mus*cs+(mub-mus)*p['B']),
        'decomposition':'mu_small*(c_large-c_small) + (mu_large-mu_small)*(c_large+V)',
        'decomposition_limit':'Algebraic accounting at a stated reference markup, not an empirical separation of mechanisms',
        'checks':{'numeric_optimizations':len(checks),'global_derivative_signs':True,
                  'analytical_numeric_crossing_match':True,'decomposition_matches_prices':True,
                  'cost_only_no_sign_change':True,'demand_only_no_sign_change':True}}
    (out/'host_portfolio_model_method.json').write_text(json.dumps(method,ensure_ascii=False,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                         'axes.spines.top':False,'axes.spines.right':False})
    f,axes=plt.subplots(1,3,figsize=(14,4.6))
    n=np.linspace(1,20,201)
    axes[0].plot(n,cost(n),color=COST,lw=2.5)
    axes[0].scatter([1,20],[cs,cb],color=COST,zorder=3)
    axes[0].set(xlabel='Объявлений у хозяина',ylabel='Издержки на занятую ночь, £',
                xticks=[1,5,10,15,20],title='Предположение: экономия на масштабе')
    prices=np.linspace(p['p0'],200,250)
    for eps,color,label in [(es,SMALL,'1 объявление: ε=3'),(eb,LARGE,'20 объявлений: ε=2')]:
        axes[1].plot(prices,100*booking_probability(prices,eps),color=color,lw=2.5,label=label)
    axes[1].set(xlabel='Цена текущему гостю, £',ylabel='Вероятность бронирования',title='Предположение: разный спрос')
    axes[1].yaxis.set_major_formatter(PercentFormatter());axes[1].legend(fontsize=9,frameon=False)
    axes[2].plot(days,V,color=SMALL,lw=2.5)
    axes[2].axhline(p['B'],color='#999',ls=':',lw=1)
    axes[2].set(xlabel='Дней до заезда',ylabel='Ожидаемая чистая прибыль ожидания, £',
                title='Общая для обоих ценность ожидания')
    for ax in axes:ax.grid(alpha=.15)
    f.text(.5,.02,'Три заданные предпосылки. Функция V(L) одинакова для обоих типов; различаются текущие издержки и спрос.\n'
           'Все параметры иллюстративные и не оценивались по данным Airbnb.',ha='center',fontsize=10)
    f.tight_layout(rect=(0,.13,1,1));save(f,fig/'host_portfolio_model_inputs')

    f,axes=plt.subplots(1,3,figsize=(15,5.2))
    for prices,color,label in [(ps,SMALL,'1 объявление'),(pb,LARGE,'20 объявлений')]:
        axes[0].plot(days,prices,color=color,lw=2.5,label=label)
    axes[0].scatter([Lstar],[90],color='#333',s=25,zorder=4)
    axes[0].set(xlabel='Дней до заезда',ylabel='Оптимальная цена, £',title='Решение задачи хозяина')
    axes[0].legend(fontsize=9,frameon=False)
    gap=pb-ps
    axes[1].plot(days,gap,color=TOTAL,lw=2.5)
    axes[1].fill_between(days,0,gap,where=gap<0,color='#BD4C42',alpha=.13)
    axes[1].fill_between(days,0,gap,where=gap>=0,color='#44835D',alpha=.13)
    axes[1].axhline(0,color='#777',lw=1)
    axes[1].axvline(Lstar,color='#999',ls=':',lw=1)
    axes[1].annotate(f'Порог: {Lstar:.2f} дня',(Lstar,0),xytext=(27,-4),fontsize=10,
                     arrowprops={'arrowstyle':'->','color':'#777'})
    axes[1].set(xlabel='Дней до заезда',ylabel='Цена крупного минус цена малого, £',
                title='Цена крупного хозяина меняет положение',ylim=(-7,17))
    axes[2].plot(days,cost_part,color=COST,lw=2,label='Экономия: −£22,50')
    axes[2].plot(days,markup_part,color=LARGE,lw=2,label='Разница наценки: £17,50 + 0,5V')
    axes[2].plot(days,gap,color=TOTAL,lw=2.5,label='Итог: −£5 + 0,5V')
    axes[2].axhline(0,color='#777',lw=1)
    axes[2].set(xlabel='Дней до заезда',ylabel='Компонент разницы цен, £',title='Разбор двух противоположных сил')
    axes[2].legend(fontsize=8.5,frameon=False,loc='lower left',bbox_to_anchor=(0,.12))
    for ax in axes:ax.grid(alpha=.15)
    f.text(.5,.02,'Общая функция V(L)=40×(1−exp(−L/30)); параметры заданы для примера.\n'
           'Разбор справа использует наценку малого хозяина как точку отсчёта. Это алгебра модели, а не оценки каналов по OLS.\n'
           'Порог 8,63 дня — результат выбранных параметров; он не оценён по Airbnb.',ha='center',fontsize=10)
    f.tight_layout(rect=(0,.15,1,1));save(f,fig/'host_portfolio_model_solution')

    f,ax=plt.subplots(figsize=(10,6.1))
    styles=[('Both mechanisms',TOTAL,'Оба механизма','-',2.7),
            ('Scale economy only',COST,'Только экономия на масштабе','--',2),
            ('Different demand only',LARGE,'Только разная чувствительность спроса','--',2),
            ('Neither mechanism','#888','Оба типа одинаковы',':',1.7)]
    for key,color,label,ls,lw in styles:
        q=scenarios[scenarios.scenario.eq(key)]
        ax.plot(q.lead_days,q.gap_gbp,color=color,label=label,ls=ls,lw=lw)
    ax.set(xlabel='Дней до заезда',ylabel='Цена крупного минус цена малого, £',
           title='В этой модели для смены знака нужны оба механизма')
    handles,labels=ax.get_legend_handles_labels()
    f.legend(handles,labels,fontsize=9,frameon=False,loc='upper center',bbox_to_anchor=(.5,.22),ncol=2)
    ax.grid(alpha=.15)
    f.text(.5,.02,'В каждом варианте меняются только указанные предпосылки; V(L) остаётся общей.\n'
           'Экономия отдельно даёт отрицательную разницу, менее чувствительный спрос отдельно — положительную.\n'
           'Это сравнение теоретических вариантов, не дополнительные результаты Airbnb.',ha='center',fontsize=10)
    f.tight_layout(rect=(0,.25,1,1));save(f,fig/'host_portfolio_model_scenarios')
    return examples,method


def save(figure,path):
    figure.savefig(str(path)+'.png',dpi=165,bbox_inches='tight')
    svg=Path(str(path)+'.svg')
    figure.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(figure)


if __name__=='__main__':
    examples,method=build()
    print(examples.round(4).to_string(index=False))
    print(json.dumps({'waiting_threshold_gbp':method['waiting_threshold_gbp'],
        'lead_threshold_days':method['lead_threshold_days'],'checks':method['checks']},indent=2))
