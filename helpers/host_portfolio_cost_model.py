"""Finite-horizon pricing with identical demand; host types differ only in cost."""
from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parents[1]
PARAMETERS={'regular_price':150.,'discount_price':100.,'high_guest_probability':.3,
    'budget_guest_probability':.3,'small_cost':70.,'large_cost':40.,'small_N':1,'large_N':20,
    'max_opportunities':30}
SMALL,LARGE,REGULAR,DISCOUNT='#176B82','#9866A4','#176B82','#D48743'
NOTEBOOK_INTRO=r'''#### 7.2.4. Простой механизм: почему скидка перед заездом выгодна крупному хозяину

В OLS связь портфеля с ценой различалась по сроку до заезда. Теперь построим небольшой теоретический пример: может ли одна экономия на масштабе объяснить, почему крупный хозяин снижает цену в последний момент, а маленький сохраняет прежнюю?

Рассматриваем ещё свободную ночь одного объекта. У всех хозяев одинаковый спрос и одинаковые возможности продать её: в каждый день с вероятностью 30% приходит гость, готовый заплатить £150, с вероятностью 30% — гость с бюджетом £100, с вероятностью 40% бронирования нет. Поэтому при цене £150 вероятность продажи равна 30%, при £100 — 60%. Публичная скидка достаётся и гостям, которые заплатили бы полную цену.

**Единственное различие хозяев — переменные издержки занятой ночи:** £70 у малого и £40 у крупного благодаря экономии на масштабе. Оба могут свободно менять цену и максимизируют ожидаемую прибыль. Цена может быть любой, но при двух уровнях готовности платить лучший вариант достигается при £100, £150 или ожидании: промежуточная цена теряет выручку без прироста спроса.

Если будущая возможность продать ночь стоит $V$, задача текущего дня:

$$\max_{p\ge0}\;q(p)(p-c)+[1-q(p)]V.$$

Первое слагаемое — прибыль от продажи сейчас, второе — прибыль от следующей возможности, если гость отказался. Решаем с конца: в день заезда $V=0$, поскольку непроданная ночь затем исчезает. Полученную максимальную прибыль используем как $V$ для предыдущего дня. Поэтому ценность ожидания вычисляется из задачи и различается между хозяевами только вследствие разных издержек. Ниже все суммы относятся к одному ещё свободному объекту.'''
NOTEBOOK_CODE='''from importlib import reload
from helpers import host_portfolio_cost_model as cost_model
cost_example, cost_method = reload(cost_model).show_notebook_example()
'''


def cost(N,p=PARAMETERS):
    return p['small_cost']-(p['small_cost']-p['large_cost'])*np.log(np.asarray(N))/np.log(p['large_N'])


def probability(price,p=PARAMETERS):
    price=np.asarray(price)
    a,b=p['high_guest_probability'],p['budget_guest_probability']
    return np.where(price<=p['discount_price'],a+b,np.where(price<=p['regular_price'],a,0.))


def solve(c,p=PARAMETERS,horizon=None):
    horizon=p['max_opportunities'] if horizon is None else horizon
    H,D=p['regular_price'],p['discount_price']
    a,b=p['high_guest_probability'],p['budget_guest_probability']
    value=0.;rows=[]
    for remaining in range(1,horizon+1):
        full=a*(H-c)+(1-a)*value
        low=(a+b)*(D-c)+(1-a-b)*value
        waiting=value
        idx=int(np.argmax([full,low,waiting]))
        next_value=max(full,low,waiting)
        rows.append({'opportunities_remaining':remaining,'days_before_checkin':remaining-1,
            'cost_gbp':c,'continuation_value_gbp':value,'regular_expected_profit_gbp':full,
            'discount_expected_profit_gbp':low,'wait_expected_profit_gbp':waiting,
            'discount_gain_gbp':low-full,'value_gbp':next_value,
            'action':['regular','discount','wait'][idx],
            'optimal_price_gbp':[H,D,np.nan][idx]})
        value=next_value
    return pd.DataFrame(rows)


def schedule_profit(actions,c,p=PARAMETERS):
    """Independent check: sum probabilities of first booking along a full price schedule."""
    survival=1.;profit=0.
    for action in actions:
        price={'regular':p['regular_price'],'discount':p['discount_price'],'wait':p['regular_price']+1}[action]
        q=float(probability(price,p))
        profit += survival*q*(price-c)
        survival *= 1-q
    return profit


def build(root=ROOT):
    root=Path(root);out,fig=root/'results',root/'figures'
    out.mkdir(exist_ok=True);fig.mkdir(exist_ok=True)
    p=PARAMETERS;H,D=p['regular_price'],p['discount_price'];a,b=p['high_guest_probability'],p['budget_guest_probability']
    assert 0<a and 0<b and a+b<1 and 0<p['large_cost']<p['small_cost']<D<H
    threshold=D-a*(H-D)/b
    curves=[]
    for N in range(1,21):
        q=solve(float(cost(N)));q['N']=N
        q['extra_budget_margin_gbp']=b*(D-q.cost_gbp)
        q['discount_to_high_guests_gbp']=a*(H-D)
        q['foregone_waiting_gbp']=b*q.continuation_value_gbp
        assert np.allclose(q.discount_gain_gbp,q.extra_budget_margin_gbp-q.discount_to_high_guests_gbp-q.foregone_waiting_gbp)
        curves.append(q)
    policies=pd.concat(curves,ignore_index=True)
    policies.to_csv(out/'host_portfolio_cost_model_policies.csv',index=False)
    examples=policies[policies.N.isin([1,20])&policies.days_before_checkin.isin([0,1,7,29])].copy()
    examples.to_csv(out/'host_portfolio_cost_model_examples.csv',index=False)

    # On each constant-demand price interval profit is increasing in price.
    # Check the global H/D/wait solution against a dense unrestricted price grid.
    price_grid=np.arange(0.,200.0001,.1)
    probs=probability(price_grid)
    for row in policies.itertuples():
        allprofits=probs*(price_grid-row.cost_gbp)+(1-probs)*row.continuation_value_gbp
        assert np.isclose(allprofits.max(),row.value_gbp,atol=1e-10)
    exhaustive=[]
    for N in [1,5,10,20]:
        c=float(cost(N));values=solve(c,horizon=4)
        for horizon in range(1,5):
            best=max(schedule_profit(s,c) for s in product(['regular','discount','wait'],repeat=horizon))
            assert np.isclose(best,values.iloc[horizon-1].value_gbp)
            exhaustive.append(dict(N=N,horizon=horizon,enumerated_schedules=3**horizon,value_gbp=best))
    pd.DataFrame(exhaustive).to_csv(out/'host_portfolio_cost_model_checks.csv',index=False)
    # Complete two-period schedules, useful for checking the backward solution by hand.
    schedule_rows=[]
    for N in [1,20]:
        for sequence in product(['regular','discount','wait'],repeat=2):
            schedule_rows.append(dict(N=N,early_action=sequence[0],same_day_action_if_unsold=sequence[1],
                expected_profit_per_listing=schedule_profit(sequence,float(cost(N)))))
    pd.DataFrame(schedule_rows).to_csv(out/'host_portfolio_cost_model_two_periods.csv',index=False)
    # Closed-form values follow because all types in this cost range charge H before the final chance.
    for N in range(1,21):
        q=policies[policies.N.eq(N)];c=float(cost(N));terminal=max(a*(H-c),(a+b)*(D-c))
        closed=H-c-(H-c-terminal)*(1-a)**(q.opportunities_remaining.to_numpy()-1)
        assert np.allclose(closed,q.value_gbp)
        assert q.loc[q.days_before_checkin.gt(0),'action'].eq('regular').all()
    small=policies[policies.N.eq(1)];large=policies[policies.N.eq(20)]
    assert small.action.eq('regular').all()
    assert large.iloc[0].action=='discount' and large.iloc[1:].action.eq('regular').all()
    today=policies[policies.days_before_checkin.eq(0)]
    assert (today.action.eq('discount')==(today.cost_gbp<threshold)).all()
    scenarios=[]
    for label,cs,cl in [('Different costs',70.,40.),('Both cost 70',70.,70.),('Both cost 40',40.,40.)]:
        for name,c in [('small',cs),('large',cl)]:
            q=solve(c);q['scenario']=label;q['type']=name;scenarios.append(q)
    scenarios=pd.concat(scenarios,ignore_index=True)
    scenarios.to_csv(out/'host_portfolio_cost_model_scenarios.csv',index=False)
    for key in ['Both cost 70','Both cost 40']:
        q=scenarios[scenarios.scenario.eq(key)]
        assert np.allclose(q[q.type.eq('small')].optimal_price_gbp,q[q.type.eq('large')].optimal_price_gbp)
    method={'status':'Illustrative proposed model; no parameters estimated or calibrated to Airbnb',
        'parameters':p,'only_host_heterogeneity':'Variable service cost c(N); risk-neutral expected profit; identical demand and opportunities',
        'cost_function':'70 - 30*ln(N)/ln(20), for N in [1,20]',
        'timing':'0 days means the final booking opportunity, then the night expires; one opportunity per day is an assumption',
        'demand':'Same at every date and for every host: high guest with probability .3, budget guest .3, no guest .4',
        'price_choice':'Any p>=0; global optimal candidates are H=150, D=100 and waiting. These are not exogenous price restrictions.',
        'discount_cannibalization':'One public price applies also to guests willing to pay H; no guest-specific price discrimination',
        'bellman':'V_0(c)=0; V_r(c)=max{a*(H-c)+(1-a)*V_(r-1), (a+b)*(D-c)+(1-a-b)*V_(r-1), V_(r-1)}',
        'discount_gain':'b*(D-c-V_(r-1)) - a*(H-D)',
        'terminal_cost_threshold':threshold,
        'waiting_threshold':'D-c - a*(H-D)/b; discount if endogenous continuation value is below this threshold',
        'portfolio_threshold_same_day':float(20**((70-threshold)/30)),
        'endogenous_waiting_value':'Differs only as a consequence of cost; no direct N-specific future-demand advantage is assumed',
        'scope':'One representative unbooked listing-night; total host portfolio N stays fixed as inventory sells. Linear expected profit makes listings separable.',
        'limitations':'Two willingness-to-pay points and one opportunity/day simplify demand and timing. No claim to explain the positive early Airbnb portfolio premium.',
        'checks':{'unrestricted_price_grid_states':len(policies),'exhaustive_schedule_comparisons':len(exhaustive),
            'enumerated_schedules':sum(x['enumerated_schedules'] for x in exhaustive),
            'closed_form_values_match':True,'discount_gain_decomposition_matches':True,
            'small_never_discounts':True,'large_discounts_only_final_day':True,'identical_costs_imply_identical_policy':True}}
    (out/'host_portfolio_cost_model_method.json').write_text(json.dumps(method,ensure_ascii=False,indent=2)+'\n')
    plots(policies,scenarios,root,threshold)
    return examples,method


def plots(policies,scenarios,root,threshold):
    fig=root/'figures';p=PARAMETERS;H,D=p['regular_price'],p['discount_price']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    f,axes=plt.subplots(1,3,figsize=(14,4.7))
    ns=np.linspace(1,20,201)
    axes[0].plot(ns,cost(ns),color=DISCOUNT,lw=2.5)
    axes[0].axhline(threshold,color='#888',ls=':',label='Порог скидки сегодня: £50')
    axes[0].set(xlabel='Объявлений у хозяина',ylabel='Затраты на занятую ночь, £',
        xticks=[1,5,10,15,20],title='Единственное различие: издержки')
    axes[0].legend(fontsize=8.5,frameon=False)
    prices=np.arange(0,181,.1)
    axes[1].plot(prices,100*probability(prices),color=REGULAR,lw=2.5)
    axes[1].set(xlabel='Цена за ночь, £',ylabel='Вероятность бронирования',title='Одинаковый спрос у всех хозяев',
        xticks=[0,50,100,150,180],ylim=(-3,66))
    axes[1].yaxis.set_major_formatter(PercentFormatter())
    for N,color,label in [(1,SMALL,'1 объявление'),(20,LARGE,'20 объявлений')]:
        q=policies[policies.N.eq(N)]
        axes[2].plot(q.days_before_checkin,q.continuation_value_gbp,color=color,lw=2.5,label=label)
    axes[2].set(xlabel='Дней до заезда',ylabel='Ценность ожидания на объект, £',title='Ожидание вычислено из задачи')
    axes[2].legend(fontsize=9,frameon=False)
    for ax in axes:ax.grid(alpha=.13)
    f.text(.5,.02,'Одинаковые гости, вероятности и возможности продать у обоих типов; отличается только стоимость обслуживания.\n'
        'Спрос и издержки заданы для иллюстрации, а ценность ожидания получена обратным решением задачи.',ha='center',fontsize=10)
    f.tight_layout(rect=(0,.13,1,1));save(f,fig/'host_portfolio_cost_model_inputs')

    f,axes=plt.subplots(1,3,figsize=(14,5))
    for N,color,label,ls in [(1,SMALL,'Малый: 1 объявление','-'),(20,LARGE,'Крупный: 20 объявлений','--')]:
        q=policies[policies.N.eq(N)&policies.days_before_checkin.le(7)]
        axes[0].step(q.days_before_checkin,q.optimal_price_gbp,where='post',color=color,lw=2.8,label=label,ls=ls)
        axes[0].scatter(q.days_before_checkin,q.optimal_price_gbp,color=color,s=15)
        axes[2].plot(q.days_before_checkin,q.discount_gain_gbp,color=color,lw=2.5,marker='o',ms=3,label=label)
    axes[0].set(xlabel='Дней до заезда (0 — сегодня)',ylabel='Оптимальная цена, £',title='Скидку даёт только крупный хозяин',
        ylim=(90,165),xticks=range(8),yticks=[100,125,150])
    axes[0].legend(fontsize=8.5,frameon=False,loc='center right')
    today=policies[policies.N.isin([1,20])&policies.days_before_checkin.eq(0)].sort_values('N')
    x=np.arange(2);w=.34
    for offset,col,color,label in [(-w/2,'regular_expected_profit_gbp',REGULAR,'Цена £150'),
                                   (w/2,'discount_expected_profit_gbp',DISCOUNT,'Цена £100')]:
        bars=axes[1].bar(x+offset,today[col],width=w,color=color,label=label)
        for bar in bars:axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.6,f'£{bar.get_height():.0f}',ha='center',fontsize=10)
    axes[1].set(xticks=x,xticklabels=['Малый','Крупный'],ylabel='Ожидаемая прибыль на объект, £',
        title='В день заезда: сравнение прибыли',ylim=(0,44))
    axes[1].legend(fontsize=9,frameon=False,loc='upper left')
    axes[2].axhline(0,color='#777',lw=1)
    axes[2].set(xlabel='Дней до заезда (0 — сегодня)',ylabel='Выигрыш от скидки вместо £150, £',
        title='Заранее скидка невыгодна обоим',xticks=range(8))
    for ax in axes:ax.grid(axis='y',alpha=.13)
    f.text(.5,.02,'Хозяин выбирает цену из всех p ≥ 0. Максимум достигается при £100 или £150 из-за двух сегментов гостей.\n'
        'Если цена не снизилась, это результат максимизации, а не запрет на скидку. Непроданная ночь после заезда даёт ноль.\n'
        'Два уровня цен и одна возможность в день — упрощения; это не оценённые цены или сроки Airbnb.',ha='center',fontsize=10)
    f.tight_layout(rect=(0,.16,1,1));save(f,fig/'host_portfolio_cost_model_solution')

    f,axes=plt.subplots(1,2,figsize=(12,5.1))
    today=policies[policies.days_before_checkin.eq(0)].sort_values('cost_gbp')
    axes[0].plot(today.cost_gbp,today.discount_gain_gbp,color=REGULAR,lw=2.5)
    axes[0].axhline(0,color='#777',lw=1);axes[0].axvline(threshold,color='#999',ls=':')
    axes[0].scatter([40,70],[3,-6],color=[LARGE,SMALL],s=45,zorder=3)
    axes[0].annotate('Крупный: +£3',(40,3),xytext=(42,2),fontsize=10)
    axes[0].annotate('Малый: −£6',(70,-6),xytext=(57,-5.7),fontsize=10)
    axes[0].set(xlabel='Переменные издержки, £',ylabel='Выигрыш от скидки в день заезда, £',
        title='Меньшие затраты делают скидку выгодной')
    cases=[('Different costs','Разные\nиздержки'),('Both cost 70','У обоих\n£70'),('Both cost 40','У обоих\n£40')]
    xx=np.arange(3);w=.34
    for offset,typ,color,label in [(-w/2,'small',SMALL,'Малый'),(w/2,'large',LARGE,'Крупный')]:
        vals=[scenarios.loc[scenarios.scenario.eq(key)&scenarios.type.eq(typ)&scenarios.days_before_checkin.eq(0),'optimal_price_gbp'].iloc[0] for key,_ in cases]
        bars=axes[1].bar(xx+offset,vals,width=w,color=color,label=label)
        for bar in bars:axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'£{bar.get_height():.0f}',ha='center',fontsize=10)
    axes[1].set(xticks=xx,xticklabels=[label for _,label in cases],ylabel='Цена в день заезда, £',
        ylim=(0,185),title='При одинаковых затратах решения совпадают')
    axes[1].legend(fontsize=9,frameon=False,loc='upper right')
    for ax in axes:ax.grid(axis='y',alpha=.13)
    f.text(.5,.02,'Справа меняются только издержки. Спрос, риск, число оставшихся попыток и правила продажи одинаковы.\n'
        'При одинаковых затратах исчезает различие между решениями малого и крупного хозяина.',ha='center',fontsize=10)
    f.tight_layout(rect=(0,.13,1,1));save(f,fig/'host_portfolio_cost_model_scenarios')


def save(figure,path):
    figure.savefig(str(path)+'.png',dpi=165,bbox_inches='tight')
    svg=Path(str(path)+'.svg');figure.savefig(svg,bbox_inches='tight')
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n');plt.close(figure)


def show_notebook_example(root=ROOT):
    """Compact worked tables for the main empirical notebook, computed from the solved model."""
    from IPython.display import display,Image,Markdown,HTML
    root=Path(root);examples,method=build(root)
    p=PARAMETERS;H,D=p['regular_price'],p['discount_price'];a,b=p['high_guest_probability'],p['budget_guest_probability']
    def number(value):
        return (f'{value:.0f}' if np.isclose(value,round(value)) else f'{value:.2f}').replace('.',',')
    today=[];early=[]
    for N,label in [(1,'Малый'),(20,'Крупный')]:
        q=examples[examples.N.eq(N)]
        t=q[q.days_before_checkin.eq(0)].iloc[0];e=q[q.days_before_checkin.eq(1)].iloc[0]
        c=t.cost_gbp;v=e.continuation_value_gbp
        today.append({'Хозяин':label,'Издержки':f'£{number(c)}',
            'При цене £150':f'{number(a)} × ({number(H)} − {number(c)}) = £{number(t.regular_expected_profit_gbp)}',
            'При цене £100':f'{number(a+b)} × ({number(D)} − {number(c)}) = £{number(t.discount_expected_profit_gbp)}',
            'Выбор цены':f'£{number(t.optimal_price_gbp)}'})
        early.append({'Хозяин':label,'Ожидание V':f'£{number(v)}',
            'При цене £150':f'{number(t.regular_expected_profit_gbp)} + {number(1-a)} × {number(v)} = £{number(e.regular_expected_profit_gbp)}',
            'При цене £100':f'{number(t.discount_expected_profit_gbp)} + {number(1-a-b)} × {number(v)} = £{number(e.discount_expected_profit_gbp)}',
            'Выбор цены':f'£{number(e.optimal_price_gbp)}'})
    today=pd.DataFrame(today);early=pd.DataFrame(early)
    today.to_csv(root/'results/host_portfolio_cost_model_worked_today.csv',index=False)
    early.to_csv(root/'results/host_portfolio_cost_model_worked_early.csv',index=False)
    display(Markdown('**В день заезда: ждать уже нельзя.** Сравниваем ожидаемую прибыль от полной цены и скидки. У крупного хозяина дополнительные гости окупают меньшую выручку с каждого бронирования; у малого — нет.'))
    display(HTML(today.to_html(index=False)))
    display(Markdown('**За день до заезда: остаётся ещё одна попытка.** Если сегодня не купят, завтра малый хозяин сможет заработать в среднем £24, крупный — £36. Поэтому к текущей прибыли добавляем ценность этой возможности с вероятностью отказа гостя: 70% при полной цене и 40% при скидке.'))
    display(HTML(early.to_html(index=False)))
    display(Markdown('Заранее оба выбирают **£150**. Когда наступает день заезда, малый сохраняет **£150**, а крупный снижает цену до **£100**. Это результат решения задачи: скидка не запрещена ни одному типу.'))
    display(Image(filename=str(root/'figures/host_portfolio_cost_model_solution.png')))
    display(Markdown(r'''**Общее условие решения.** Выигрыш от скидки вместо полной цены:

$$\Delta\Pi=0{,}3(100-c)-0{,}3(150-100)-0{,}3V=0{,}3(50-c-V).$$

Первый член — прибыль от дополнительных бюджетных гостей; второй — потеря выручки от тех, кто заплатил бы полную цену; третий — потеря будущей возможности для дополнительно проданных ночей. Скидка выгодна, если **$c+V<£50$**. В день заезда $V=0$: у крупного издержки £40, условие выполняется; у малого £70, не выполняется. За день до заезда у крупного $c+V=40+36=76$, поэтому он тоже сохраняет полную цену. При одинаковых издержках решения двух хозяев совпадают.

**Что это даёт проекту.** Модель показывает возможный механизм разных скидок перед заездом при одинаковом спросе. Положительную премию крупного хозяина заранее она не объясняет: в этом примере заранее оба ставят £150. Все числа и одна возможность продажи в день заданы для иллюстрации; OLS не оценивала эти издержки и не наблюдала изменения цены одного объекта во времени. Поэтому это теоретическое объяснение для дальнейшей проверки. Полный вывод, три графика и проверки решения находятся в [отдельном теоретическом ноутбуке](host_portfolio_economic_model.ipynb).'''))
    return examples,method


if __name__=='__main__':
    examples,method=build()
    print(examples[['N','days_before_checkin','cost_gbp','continuation_value_gbp',
        'regular_expected_profit_gbp','discount_expected_profit_gbp','optimal_price_gbp']].to_string(index=False))
    print(json.dumps(method['checks'],indent=2))
