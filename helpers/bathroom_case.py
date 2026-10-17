"""Exploratory bathroom/room-type finding, reproduced with the fitted main models.

No model is fitted here. SHAP interactions explain the model; synthetic response
contrasts are not estimates of a causal price effect. Selection was exploratory.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import DMatrix

ROOMS = ['Entire home/apt', 'Private room']
LABELS = ['Целая квартира', 'Отдельная комната']
COLORS = ['#176B82', '#D48743']


def bathroom_case(d, X, pre, xgb, ols, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    out, fig = root / 'results', root / 'figures'
    # Reuse exactly the sample from section 7.2, including its row order.
    sample = pd.read_csv(out / 'shap_sample.csv')
    positions = pd.Index(d.id).get_indexer(sample.id)
    assert (positions >= 0).all() and d.id.is_unique
    sx = X.iloc[positions]
    matrix = pre.transform(sx)
    prediction = xgb.predict(matrix)
    assert np.allclose(prediction, sample.prediction_log, atol=2e-5)
    interaction = xgb.get_booster().predict(DMatrix(matrix), pred_interactions=True)
    encoded = pre.get_feature_names_out()
    raw = list(pre.feature_names_in_)
    mapping = []
    for name in encoded:
        suffix = name.split('__', 1)[1].removeprefix('missingindicator_')
        mapping.append(next(c for c in sorted(raw, key=len, reverse=True)
                            if suffix == c or suffix.startswith(c + '_')))
    columns = list(dict.fromkeys(mapping))
    membership = np.array([[m == c for c in columns] for m in mapping], dtype=np.float32)
    aggregated = np.einsum('nia,ib->nba', interaction[:, :-1, :-1], membership, optimize=True)
    aggregated = np.einsum('nia,ab->nib', aggregated, membership, optimize=True)
    saved = pd.read_csv(out / 'shap_contributions.csv')
    assert saved.id.equals(sample.id)
    assert np.allclose(aggregated.sum(axis=2), saved[columns], atol=2e-5)
    ranks = []
    for i, a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            ranks.append({'feature_a': a, 'feature_b': columns[j],
                          'mean_abs_full_pair': float(np.abs(2 * aggregated[:, i, j]).mean())})
    ranking = pd.DataFrame(ranks).sort_values('mean_abs_full_pair', ascending=False).reset_index(drop=True)
    ranking.to_csv(out / 'bathroom_interaction_ranking.csv', index=False)
    i, j = columns.index('bathroom_count'), columns.index('room_type')
    pair = 2 * aggregated[:, i, j]  # symmetric matrix stores half in each cell
    points = pd.DataFrame({'id': sample.id, 'room_type': sx.room_type.to_numpy(),
                           'bathroom_count': sx.bathroom_count.to_numpy(),
                           'bathroom_shap': saved.bathroom_count,
                           'full_pair_shap': pair})
    points.to_csv(out / 'bathroom_shap_pair_sample.csv', index=False)
    rng = np.random.default_rng(20261017)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    f, ax = plt.subplots(figsize=(9, 4.8))
    for offset, room, label, color in zip([-.15, .15], ROOMS, LABELS, COLORS):
        for k, count in enumerate([1., 1.5, 2.]):
            v = points.loc[points.room_type.eq(room) & points.bathroom_count.eq(count), 'full_pair_shap']
            ax.scatter(k + offset + rng.uniform(-.065, .065, len(v)), v,
                       s=16, alpha=.35, color=color, edgecolors='none',
                       label=label if k == 0 else None)
            ax.scatter(k + offset, v.mean(), marker='D', s=55, color=color,
                       edgecolors='black', linewidths=.6, zorder=3)
            ax.annotate(f'n={len(v)}', (k + offset, v.min()), xytext=(0, -18),
                        textcoords='offset points', ha='center', fontsize=9, color=color)
    ax.axhline(0, color='#777777', lw=.8)
    ax.set(xticks=[0, 1, 2], xticklabels=['1', '1,5', '2'],
           xlabel='Наблюдаемое число ванных', ylabel='Парный SHAP-вклад, log(цены)',
           title='Взаимодействие числа ванных и типа жилья')
    ax.legend(loc='lower left', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
    f.text(.5, .01, 'Точка — объявление; ромб — среднее. Показаны 898 объектов из исходных 1 000.\n'
           'Знак относится к взаимодействию, а не ко всему вкладу числа ванных.', ha='center', fontsize=9)
    f.tight_layout(rect=(0, .09, 1, 1)); f.savefig(fig / 'bathroom_shap_interaction.png', dpi=160, bbox_inches='tight'); plt.close(f)
    # One-input contrasts on the two already examined evaluation splits.
    rows, individual = [], []
    eligible = X.room_type.isin(ROOMS) & X.bathroom_count.between(1, 2)
    for split in ['validation', 'holdout']:
        mask = d.split.eq(split) & eligible
        lo, hi = X.loc[mask].copy(), X.loc[mask].copy()
        lo['bathroom_count'], hi['bathroom_count'] = 1., 2.
        low, high = pre.transform(lo), pre.transform(hi)
        delta = xgb.predict(high) - xgb.predict(low)
        linear_delta = ols.predict(high) - ols.predict(low)
        z = pd.DataFrame({'id': d.loc[mask, 'id'].to_numpy(), 'split': split,
                          'room_type': lo.room_type.to_numpy(),
                          'bathroom_shared': lo.bathroom_shared.to_numpy(),
                          'xgb_delta_log': delta, 'ols_delta_log': linear_delta})
        individual.append(z)
        for room, label in zip(ROOMS, LABELS):
            g = z[z.room_type.eq(room)]
            rows.append({'split': split, 'room_type': room, 'n': len(g),
                         'mean_delta_log': g.xgb_delta_log.mean(),
                         'xgb_change_pct': 100 * np.expm1(g.xgb_delta_log.mean()),
                         'ols_change_pct': 100 * np.expm1(g.ols_delta_log.mean()),
                         'xgb_positive_share': g.xgb_delta_log.gt(0).mean()})
    response = pd.DataFrame(rows)
    response.to_csv(out / 'bathroom_case_response.csv', index=False)
    pd.concat(individual).to_csv(out / 'bathroom_case_individual.csv', index=False)
    support = d[['id', 'split']].assign(room_type=X.room_type.to_numpy(), bathroom_count=X.bathroom_count.to_numpy())
    support = support[support.room_type.isin(ROOMS) & support.bathroom_count.isin([1., 2.])]
    support = support.groupby(['split', 'room_type', 'bathroom_count']).size().rename('n').reset_index()
    support.to_csv(out / 'bathroom_case_support.csv', index=False)
    f, ax = plt.subplots(figsize=(9, 4.7))
    for offset, split, color, label in [(-.17, 'validation', '#91BEC8', 'Validation'),
                                       (.17, 'holdout', '#176B82', 'Test')]:
        t = response[response.split.eq(split)].set_index('room_type')
        values = [t.loc[r, 'xgb_change_pct'] for r in ROOMS]
        bars = ax.bar(np.arange(2) + offset, values, width=.3, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:+.2f}%', (bar.get_x() + bar.get_width()/2, value),
                        xytext=(0, 5 if value >= 0 else -15), textcoords='offset points', ha='center')
    linear = float(response.ols_change_pct.mean())
    assert np.ptp(response.ols_change_pct) < 1e-7
    ax.axhline(linear, ls='--', color='#D48743', label=f'OLS: {linear:+.2f}% для обоих типов')
    ax.axhline(0, color='#777777', lw=.8)
    ax.set(xticks=[0, 1], xticklabels=LABELS, ylim=(-5, 25), ylabel='Изменение модельного прогноза, %',
           title='Одна дополнительная ванная: одинаковая замена, разный ответ')
    ax.legend(loc='upper right', fontsize=10); ax.spines[['top', 'right']].set_visible(False)
    f.text(.5, .01, 'Число ванных заменено с 1 на 2; остальные входы сохранены.\n'
           'Синтетический ответ модели, не причинное изменение цены.', ha='center', fontsize=9)
    f.tight_layout(rect=(0, .09, 1, 1)); f.savefig(fig / 'bathroom_model_response.png', dpi=160, bbox_inches='tight'); plt.close(f)
    metadata = {'status': 'exploratory, selected after looking at existing test data',
                'method': 'Exact native TreeSHAP interactions; symmetric entries summed for a full pair',
                'aggregation': 'Sum over encoded columns. Not a separately computed grouped Shapley game.',
                'shap_sample': 'Same 1000 rows as section 7.2; contribution sums checked',
                'response_sample': 'Existing validation/test, original bathroom_count in [1,2], entire homes/private rooms',
                'contrast': 'bathroom_count 1->2, all other inputs including bathroom_shared fixed',
                'percentage': '100 * expm1(mean delta log prediction)',
                'training': 'No models or preprocessing fitted here',
                'interpretation': 'Model behaviour; economic mechanism unproved; no significance test'}
    (out / 'bathroom_case_method.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return response, support
