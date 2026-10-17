"""Exploratory ICE-to-OLS case; original fitted prediction models stay fixed."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import linalg, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from .ols_core import estimate, contrast

GROUPS = ['Сегодня', '1–7 дней', '8–29 дней', '30+ дней']
COLORS = ['#BD4C42', '#D49B42', '#3F84A8', '#44835D']
GRID = [1, 2, 5, 10, 20]
INTRO = '''#### 7.2.3. От ICE к гипотезе: портфель хозяина и близость заезда

В обзоре ICE у числа объявлений хозяина средняя линия немного растёт, но часть индивидуальных линий идёт вниз. Начнём с этой зацепки: для каких объявлений модель снижает прогноз и что у них общего?

На первом графике повторяем панель только для `host_listings_count`. Подставляем 1, 2, 5, 10 и 20 объявлений у хозяина, сохраняя остальные входы. Каждая линия начинается с нуля при одном объявлении. Линия вниз означает снижение прогноза цены, а не улучшение точности.'''
CELL_CODE = '''from helpers.host_portfolio_case import show_portfolio_case
portfolio_ols, portfolio_ice = show_portfolio_case(
    d, X, pre, models['XGBoost'][1])
'''


def lead_groups(lead):
    lead = np.asarray(lead, dtype=float)
    return np.select([lead == 0, (lead >= 1) & (lead <= 7),
                      (lead >= 8) & (lead <= 29), lead >= 30], [0, 1, 2, 3], -1)


def eligible_rows(X):
    return (X.room_type.isin(['Entire home/apt', 'Private room'])
            & X.host_listings_count.between(1, 20)
            & X.quote_lead_days.ge(0)).to_numpy()


def portfolio_ice(d, X, pre, xgb, root):
    out, fig = root / 'results', root / 'figures'
    curves = pd.read_csv(out / 'ice_overview_curves.csv')
    curves = curves[curves.feature.eq('host_listings_count')].copy()
    endpoint = curves[curves.grid_value.eq(20)].copy()
    pos = pd.Index(d.id).get_indexer(endpoint.id)
    assert (pos >= 0).all() and d.iloc[pos].split.eq('holdout').all()
    endpoint['lead'] = X.iloc[pos].quote_lead_days.to_numpy()
    neg = endpoint.change_pct.lt(-.5)
    initial = dict(n=len(endpoint), positive=int(endpoint.change_pct.gt(.5).sum()),
                   negative=int(neg.sum()),
                   negative_near=int((neg & endpoint.lead.between(0, 7)).sum()))
    initial['near_zero'] = initial['n'] - initial['positive'] - initial['negative']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    f, ax = plt.subplots(figsize=(10, 4.8))
    rng = np.random.default_rng(20261022)
    ids = rng.choice(endpoint.id.to_numpy(), size=min(120, len(endpoint)), replace=False)
    for ident in ids:
        q = curves[curves.id.eq(ident)].sort_values('grid_value')
        ax.plot(q.grid_value, q.change_pct, color='#6F7B84', alpha=.15, lw=.8)
    mean = curves.groupby('grid_value')[['delta_log', 'ols_delta_log']].mean().loc[GRID]
    ax.plot(GRID, 100*np.expm1(mean.delta_log), color='#176B82', lw=2.5, label='XGBoost: сводная линия')
    ax.plot(GRID, 100*np.expm1(mean.ols_delta_log), color='#D48743', lw=2, ls='--', label='Исходная OLS: общий ответ')
    ax.axhline(0, color='#888', lw=.8)
    ax.set(xticks=GRID, xlabel='Число объявлений у хозяина', ylabel='Изменение прогноза цены, %',
           title=f'Зацепка: часть ICE-линий идёт вниз · n={len(endpoint)}')
    ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([0], [0], color='#6F7B84', alpha=.5, label='Отдельные объявления XGBoost')]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles+h, fontsize=9, loc='upper left')
    f.text(.5, .02, f'При замене 1 → 20: выше +0,5% — {initial["positive"]}; ниже −0,5% — {initial["negative"]}; '
           f'между ними — {initial["near_zero"]}.\nПоказаны 120 индивидуальных линий; сводные рассчитаны по всем {initial["n"]}. Остальные входы фиксированы.',
           ha='center', fontsize=9)
    f.tight_layout(rect=(0, .1, 1, 1)); f.savefig(fig/'host_portfolio_ice.png', dpi=150); plt.close(f)

    # Same contrast for all eligible rows of the existing validation and test.
    rows = []
    for split in ['validation', 'holdout']:
        mask = eligible_rows(X) & d.split.eq(split).to_numpy()
        q = X.loc[mask].copy()
        groups = lead_groups(q.quote_lead_days)
        assert (groups >= 0).all()
        base = q.copy(); base['host_listings_count'] = 1.
        p0 = xgb.predict(pre.transform(base))
        for value in GRID:
            alt = q.copy(); alt['host_listings_count'] = float(value)
            assert alt.drop(columns='host_listings_count').equals(q.drop(columns='host_listings_count'))
            delta = xgb.predict(pre.transform(alt)) - p0
            rows.append(pd.DataFrame({'id': d.loc[mask, 'id'].to_numpy(),
                'host_id': d.loc[mask, 'host_id'].to_numpy(), 'split': split,
                'group': groups, 'grid': value, 'delta_log': delta,
                'change_pct': 100*np.expm1(delta)}))
    individual = pd.concat(rows, ignore_index=True)
    individual.to_csv(out/'host_portfolio_ice_curves.csv.gz', index=False, compression={'method':'gzip','mtime':0})
    summary = []
    for (split, group, value), q in individual.groupby(['split', 'group', 'grid']):
        summary.append(dict(split=split, group=int(group), label=GROUPS[group], grid=value,
            n=len(q), hosts=q.host_id.nunique(), mean_delta_log=q.delta_log.mean(),
            change_pct=100*np.expm1(q.delta_log.mean()), negative_share=q.change_pct.lt(-.5).mean(),
            positive_share=q.change_pct.gt(.5).mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out/'host_portfolio_ice_summary.csv', index=False)
    test = summary[summary.split.eq('holdout')]
    f, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for i, color in enumerate(COLORS):
        q = test[test.group.eq(i)].sort_values('grid')
        axes[0].plot(q.grid, q.change_pct, marker='o', lw=2, color=color,
                     label=f'{GROUPS[i]} · n={q.n.iloc[0]:,}'.replace(',', ' '))
    axes[0].axhline(0, color='#888', lw=.8)
    axes[0].set(xticks=GRID, xlabel='Число объявлений у хозяина', ylabel='Изменение прогноза цены, %',
                title='Ответ XGBoost зависит от срока до заезда')
    axes[0].legend(fontsize=9)
    end = test[test.grid.eq(20)].sort_values('group')
    values = 100*end.negative_share.to_numpy()
    bars = axes[1].bar(range(4), values, color=COLORS, width=.65)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x()+bar.get_width()/2, value+1.2, f'{value:.1f}%', ha='center')
    axes[1].set(xticks=range(4), xticklabels=['Сегодня', '1–7\nдней', '8–29\nдней', '30+\nдней'],
                ylim=(0, 75), ylabel='Доля снижений более 0,5%, %', title='Где чаще встречаются линии вниз')
    for ax in axes: ax.spines[['top', 'right']].set_visible(False)
    f.text(.5, .02, 'Прежний test: 9 381 подходящее объявление. Замена 1 → 20; остальные входы фиксированы.\n'
           'Сводные линии: 100 × expm1(средняя Δlog-прогноза). «Сегодня» — дата сбора цены.\n'
           'Это поведение обученной модели; интервалы статистической точности здесь не рассчитываются.', ha='center', fontsize=9)
    f.tight_layout(rect=(0, .13, 1, 1)); f.savefig(fig/'host_portfolio_lead.png', dpi=150); plt.close(f)
    return summary, initial


def portfolio_ols(d, X, pre, root):
    assert len(d) == len(X) and not d.host_id.isna().any()
    count = X.host_listings_count.to_numpy(float)
    group = lead_groups(X.quote_lead_days)
    eligible = eligible_rows(X)
    G = np.column_stack([group == i for i in range(4)]).astype(float)
    matrix = pre.transform(X)
    matrix = matrix.toarray() if hasattr(matrix, 'toarray') else np.asarray(matrix)
    keep = pre.get_feature_names_out() != 'num__host_listings_count'
    W = np.column_stack([np.ones(len(X)), matrix[:, keep], G[:, 1:]])
    y = np.log(d.price_gbp.to_numpy(float))
    results, diagnostics, support = [], [], []
    for form, sample in [('raw_count','all'), ('raw_count','train'), ('raw_count','validation'),
                         ('raw_count','holdout'), ('log_count','all')]:
        idx = np.flatnonzero(eligible & (d.split.eq(sample).to_numpy() if sample != 'all' else True))
        assert len(idx) > 0 and (group[idx] >= 0).all()
        z = count if form == 'raw_count' else np.log(np.maximum(count, 1))
        fit = estimate(W, G*z[:,None], idx, d, y, direct_check=(form == 'raw_count' and sample == 'all'))
        scale = 19. if form == 'raw_count' else np.log(20.)
        meta = dict(form=form, sample=sample, n=fit['n'], hosts=fit['hosts'], rank=fit['rank'])
        for i, label in enumerate(GROUPS):
            results.append({**meta, 'group':i, 'label':label, **contrast(fit, np.eye(4)[i]*scale)})
        results.append({**meta, 'group':4, 'label':'30+ минус сегодня',
                        **contrast(fit, np.array([-1,0,0,1])*scale)})
        C = np.column_stack([-np.ones(3), np.eye(3)])
        b = C@fit['beta']; cov = C@fit['covariance']@C.T
        F = float(b@linalg.solve(cov,b)/3)
        diagnostics.append({**meta, 'beta':fit['beta'].tolist(), 'covariance':fit['covariance'].tolist(),
            'equal_slopes_F':F, 'df_num':3, 'df_den':fit['df'],
            'equal_slopes_p':float(stats.f.sf(F,3,fit['df'])),
            'control_orthogonality':fit['control_orthogonality'], 'focus_orthogonality':fit['focus_orthogonality']})
        if form == 'raw_count':
            for i in range(4):
                selected = idx[group[idx] == i]
                support.append({**meta, 'group':i, 'label':GROUPS[i], 'group_n':len(selected),
                                'group_hosts':d.iloc[selected].host_id.nunique()})
    results = pd.DataFrame(results)
    results.to_csv(root/'results/host_portfolio_ols.csv', index=False)
    pd.DataFrame(support).to_csv(root/'results/host_portfolio_ols_support.csv', index=False)
    method = {'status':'Exploratory hypothesis found after inspecting existing ICE/test; no new independent test',
        'outcome':'log(price_gbp)', 'cohort':'Entire home/apt or Private room; observed host_listings_count 1–20; quote_lead_days >= 0',
        'specification':'Four count-by-lead-group slopes; group main effects; all other main-model controls including numeric lead',
        'preprocessing':'Original train-fitted preprocessing; host_listings_count column replaced; SVD handles aliased controls',
        'remaining_encoded_controls':pre.get_feature_names_out()[keep].tolist(),
        'contrast':'1 to 20 listings: 19*b for raw count; log(20)*b for log count',
        'percent':'100*expm1(log contrast); exp conditional mean log price, not arithmetic mean GBP',
        'uncertainty':'CR1 cluster covariance by host_id; t(df=clusters-1), two-sided; joint Wald F',
        'checks':'Primary coefficients and clustered covariance verified against direct full OLS; residual orthogonality checked for every fit',
        'limitations':'Cross-sectional conditional associations, no within-listing price changes, prior hypothesis search not included in p-values',
        'fits':diagnostics}
    (root/'results/host_portfolio_ols_method.json').write_text(json.dumps(method, ensure_ascii=False, indent=2)+'\n')
    f, axes = plt.subplots(1,2,figsize=(12,5.2),sharex=True,sharey=True)
    for ax, sample, color, title in zip(axes,['all','holdout'],['#176B82','#D48743'],['Вся подходящая выборка','Прежний test отдельно']):
        q = results[(results.form == 'raw_count') & (results['sample'] == sample) & (results.group < 4)].sort_values('group')
        ax.errorbar(q['pct_change'], q.group, xerr=[q['pct_change']-q.pct_ci_low, q.pct_ci_high-q['pct_change']],
                    fmt='o', color=color, capsize=4, ms=7, lw=1.7)
        for row in q.itertuples():
            ax.annotate(f'{row.pct_change:+.1f}%',(row.pct_change,row.group),xytext=(0,10),
                        textcoords='offset points',ha='center',fontsize=11,color=color)
        ax.axvline(0,color='#777',lw=1)
        ax.set(yticks=range(4),yticklabels=GROUPS,xlim=(-40,40),ylim=(3.5,-.6),
               xlabel='Оценённая разница цены, 1 → 20 объявлений',
               title=f'{title}\nn={q.n.iloc[0]:,}; хозяев={q.hosts.iloc[0]:,}'.replace(',',' '))
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',alpha=.15)
    f.suptitle('OLS: связь с ценой меняет знак в зависимости от срока до заезда',fontsize=15,y=1.02)
    f.text(.5,.02,'Точки — оценки с остальными контролями; отрезки — 95% интервалы с группировкой ошибок по хозяину.\n'
           'Если отрезок пересекает ноль, направление связи определено неточно.\n'
           'Это условные различия между объявлениями и хозяевами, а не доказанные эффекты расширения портфеля.',ha='center',fontsize=9)
    f.tight_layout(rect=(0,.14,1,1));f.savefig(root/'figures/host_portfolio_ols.png',dpi=150,bbox_inches='tight');plt.close(f)
    return results


def show_portfolio_case(d, X, pre, xgb, root=None):
    from IPython.display import display, Image, Markdown
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    def text(s): display(Markdown(s))
    def image(name): display(Image(filename=str(root/'figures'/f'{name}.png')))
    ice, initial = portfolio_ice(d,X,pre,xgb,root)
    image('host_portfolio_ice')
    text(f'''**Где спрятались линии вниз?** Из {initial['n']} объявлений у {initial['negative']} прогноз падает больше чем на 0,5% при замене 1 → 20. У {initial['negative_near']} из этих {initial['negative']} ({100*initial['negative_near']/initial['negative']:.1f}%) заезд приходится на ближайшие семь дней, включая день сбора цены. Это зацепка для дальнейшего разбора.

Теперь берём весь прежний test и разделяем его по сроку до заезда: сегодня, через 1–7, 8–29 и 30+ дней. Оставляем целые квартиры и отдельные комнаты с наблюдаемым портфелем 1–20 и неотрицательным сроком: 9 381 объявление. Поэтому это уже более широкая выборка, чем исходные 800 ICE-линий. Для каждого объекта выполняем ту же замену только числа объявлений.''')
    image('host_portfolio_lead')
    end = ice[ice.split.eq('holdout') & ice.grid.eq(20)].set_index('group')
    text(f'''При заезде сегодня сводный ответ XGBoost равен **{end.loc[0,'change_pct']:+.2f}%**, а при 30+ днях — **{end.loc[3,'change_pct']:+.2f}%**. Доля заметных снижений падает с {100*end.loc[0,'negative_share']:.1f}% до {100*end.loc[3,'negative_share']:.1f}%. На прежней validation рисунок похож. Сводная линия усредняет изменения log-прогноза и затем переводит их в проценты; она не является эффектом для каждого объекта.

**Экономическая гипотеза.** Непроданную ночь нельзя продать позже. Поэтому хозяева с большим портфелем могут активнее снижать цену перед заездом, чтобы избежать простоя, а на далёкие даты удерживать более высокую цену. Проверяем наблюдаемое следствие: связь числа объявлений с ценой должна быть более отрицательной на ближайшие даты и более положительной на дальние. Саму стратегию и издержки мы не наблюдаем.

**Как проверяем через OLS.** Объясняем log цены на всех подходящих объявлениях. Сохраняем остальные контроли основной модели: размер и тип жилья, удобства, расположение, отзывы, свойства хозяина и даты предложения. Добавляем группы срока и разрешаем числу объявлений иметь свой наклон в каждой группе. В формуле это произведения `число объявлений × индикатор группы срока`. Например, одно такое произведение равно числу объявлений только для заезда сегодня, а для остальных строк равно нулю.

Точность считаем с группировкой ошибок по хозяину: несколько его объявлений не считаются независимыми. На следующем графике коэффициенты переведены в ту же разницу **1 → 20 объявлений**, при фиксированных остальных характеристиках. Точка показывает оценку, отрезок — 95% интервал. Пересечение нуля означает, что по этой оценке нельзя уверенно установить знак связи.''')
    ols = portfolio_ols(d,X,pre,root)
    image('host_portfolio_ols')
    q = ols[(ols.form == 'raw_count') & (ols['sample'] == 'all')].set_index('group')
    t = ols[(ols.form == 'raw_count') & (ols['sample'] == 'holdout')].set_index('group')
    log = ols[(ols.form == 'log_count') & (ols['sample'] == 'all')].set_index('group')
    table = ['| До заезда | Разница цены по OLS | 95% интервал |','|---|---:|---:|']
    for i,label in enumerate(GROUPS):
        row=q.loc[i];table.append(f"| {label} | {row['pct_change']:+.1f}% | от {row.pct_ci_low:+.1f}% до {row.pct_ci_high:+.1f}% |")
    text('**Что получилось в OLS.** Основная выборка: '+f"{int(q.loc[0,'n']):,} объявлений у {int(q.loc[0,'hosts']):,} хозяев.\n\n".replace(',',' ')+'\n'.join(table))
    text(f'''На всей выборке связи отрицательны для ближайших дат и положительны для дальних. Прямой тест разницы наклонов «30+ дней минус сегодня» даёт **p < 0,001**: мы проверяем само различие, а не сравниваем два отдельных p-value.

**Что выдержало проверку на прежнем test.** Знаки те же, но оценка для сегодняшнего заезда неточная: {t.loc[0,'pct_change']:+.1f}%, интервал от {t.loc[0,'pct_ci_low']:+.1f}% до {t.loc[0,'pct_ci_high']:+.1f}%. Для 30+ дней связь положительная: {t.loc[3,'pct_change']:+.1f}%, интервал от {t.loc[3,'pct_ci_low']:+.1f}% до {t.loc[3,'pct_ci_high']:+.1f}%. Разница наклонов значима, p = {t.loc[4,'p_two_sided']:.3f}. Та же форма знаков получилась отдельно на train и validation. Если заменить число объявлений его логарифмом, на всей выборке знак тоже меняется: от {log.loc[0,'pct_change']:+.1f}% сегодня до {log.loc[3,'pct_change']:+.1f}% на дальние даты; размер оценки зависит от формы модели.

**Вывод.** OLS поддерживает зависимость связи от срока до заезда. На всей выборке поддерживается и смена знака. На старом test отдельно подтверждены различие наклонов и положительная связь для дальних дат; отрицательная связь для сегодняшнего заезда остаётся неопределённой.

Это пример того, как интерпретация бустинга подсказала взаимодействие для линейной модели. Улучшение точности прогноза от этого взаимодействия здесь не проверялось. Масштабы ICE и OLS сравнивать напрямую нельзя: первый описывает отклик фиксированного XGBoost, второй — связь с наблюдаемой ценой в заново оценённой линейной модели.

**Предел вывода.** Данные согласуются с версией о скидках перед заездом, но не доказывают её. Мы сравниваем разных хозяев и объявления; изменений цены одного объекта во времени здесь нет. Доступность жилья, механизм сбора предложений и ненаблюдаемые различия могут объяснять часть связи. Анализ ограничен портфелями 1–20; на дальних датах в train лишь 20 строк с ровно 20 объявлениями у хозяина, поэтому крайнее ICE-сравнение имеет слабую поддержку. Это число строк, а не независимых хозяев. `host_listings_count` также не измеряет конкуренцию в районе.

Гипотеза найдена после просмотра ICE, а все части данных уже изучались. Это разведочная проверка; p-value не учитывают предшествующий поиск гипотез. В таблицах проценты рассчитаны как `100 × (exp(19 × наклон) − 1)`: строго они относятся к экспоненте среднего log цены, а не к арифметической средней цене в фунтах. Все оценки, интервалы и сведения о расчёте сохранены в `results/host_portfolio_ols*`, ответы XGBoost — в `results/host_portfolio_ice*`.''')
    return ols, ice
