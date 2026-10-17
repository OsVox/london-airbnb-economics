"""Compare one host-disjoint validation split with existing grouped CV results.

Reads calculated scores; does not refit or tune a model. The one-split baseline
is validation (not holdout) and matches the fixed CV hyperparameters exactly.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]


def plot_split_comparison(root=None):
    root = Path(root) if root is not None else ROOT
    cv = pd.read_csv(root / 'results/development_cv.csv')
    tuning = pd.read_csv(root / 'results/tuning.csv')
    audit = json.loads((root / 'results/audit.json').read_text())
    baseline_train_n = int(audit['splits']['train']['rows'])
    baseline_validation_n = int(audit['splits']['validation']['rows'])
    settings = {'Ridge': ('alpha', 10), 'XGBoost': ('max_depth', 5)}
    rows = []
    for model, (parameter, value) in settings.items():
        baseline = tuning.loc[
            tuning.model.eq(model)
            & tuning.params.map(lambda text: json.loads(text).get(parameter) == value)
        ]
        assert len(baseline) == 1
        folds = cv.loc[cv.model.eq(model)].sort_values('fold')
        assert folds.fold.tolist() == [1, 2, 3]
        for metric in ['log_mae', 'gbp_mae']:
            single = float(baseline.iloc[0][metric])
            mean = float(folds[metric].mean())
            rows.append(dict(
                model=model, parameter=parameter, value=value, metric=metric,
                single_validation=single, grouped_cv_mean=mean,
                grouped_cv_min=float(folds[metric].min()),
                grouped_cv_max=float(folds[metric].max()),
                error_increase_pct=100 * (mean / single - 1),
                single_n_train=baseline_train_n,
                single_n_validation=baseline_validation_n,
                cv_n_train_min=int(folds.n_train.min()),
                cv_n_train_max=int(folds.n_train.max()),
                cv_n_validation_min=int(folds.n_validation.min()),
                cv_n_validation_max=int(folds.n_validation.max()),
            ))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(root / 'results/split_vs_groupkfold.csv', index=False)
    previous_style = plt.rcParams.copy()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.3), sharey=True)
    blue, orange, ink = '#176B82', '#E28B36', '#173442'
    for ax, (model, (parameter, value)) in zip(axes, settings.items()):
        row = comparison.loc[comparison.model.eq(model) & comparison.metric.eq('log_mae')].iloc[0]
        folds = cv.loc[cv.model.eq(model)].sort_values('fold')
        values = [row.single_validation, row.grouped_cv_mean]
        ax.bar([0, 1], values, width=.56, color=[blue, orange], zorder=2)
        for x, y in enumerate(values):
            ax.text(x, y + .008, f'{y:.4f}'.replace('.', ','), ha='center', weight='bold', color=ink)
        ax.scatter(1 + np.linspace(-.12, .12, 3), folds.log_mae,
                   s=38, color=ink, edgecolor='white', linewidth=.6, zorder=4)
        label = 'α = 10' if model == 'Ridge' else 'depth = 5'
        ax.set_title(f'{model}: {label}\nРост средней ошибки: +{row.error_increase_pct:.2f}%'.replace('.', ','),
                     fontsize=14, pad=15)
        ax.set_xticks([0, 1], ['Одно разбиение\ntrain → validation', 'GroupKFold\nсреднее 3 частей'])
        ax.set_ylim(0, .33)
        ax.set_xlim(-.6, 1.6)
        ax.grid(axis='y', alpha=.18, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('MAE логарифма цены — меньше значит лучше')
    fig.suptitle('Одно разбиение и GroupKFold: ухудшение небольшое',
                 fontsize=17, weight='bold', y=.97, color=ink)
    fig.legend(handles=[Line2D([0], [0], marker='o', color='none', markerfacecolor=ink,
                              markersize=6, label='Ошибка отдельной GroupKFold-части')],
               loc='lower center', bbox_to_anchor=(.5, .16), frameon=False, fontsize=11)
    fig.text(.5, .105,
             'Оба способа разделяют данные по хозяевам. Одинаковые признаки и настройки моделей.\n'
             f'Обучение: {baseline_train_n:,} объявление при одном разбиении; 31 820–31 821 в каждой CV-части.'.replace(',', ' '),
             ha='center', va='center', fontsize=10, color='#526772')
    fig.text(.5, .035,
             'Это чувствительность к разбиению и объёму обучения; разница не доказывает переобучение.\n'
             'Три точки показывают разброс частей, а не доверительный интервал.',
             ha='center', va='center', fontsize=10, color='#526772')
    fig.subplots_adjust(top=.75, bottom=.31, left=.085, right=.98, wspace=.16)
    (root / 'figures').mkdir(exist_ok=True)
    fig.savefig(root / 'figures/split_vs_groupkfold.png', dpi=180, bbox_inches='tight', facecolor='white')
    fig.savefig(root / 'figures/split_vs_groupkfold.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    plt.rcParams.update(previous_style)
    return comparison


if __name__ == '__main__':
    print(plot_split_comparison().to_string(index=False))
