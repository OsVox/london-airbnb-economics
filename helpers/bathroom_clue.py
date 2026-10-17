"""Isolate the bathroom_count row from the existing SHAP summary, using its data."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def bathroom_clue(d, X, root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    sample = pd.read_csv(root / 'results/shap_sample.csv')
    saved = pd.read_csv(root / 'results/shap_contributions.csv')
    assert sample.id.equals(saved.id)
    positions = pd.Index(d.id).get_indexer(sample.id)
    assert (positions >= 0).all()
    values = pd.to_numeric(X.iloc[positions].bathroom_count, errors='coerce').to_numpy()
    lo, hi = np.nanquantile(values, [.05, .95])
    normalized = np.nan_to_num(np.clip((values-lo)/max(hi-lo, 1e-9), 0, 1), nan=.5)
    # Recover the exact vertical jitter used for this row in shap_analysis.
    importance = saved.drop(columns='id').abs().mean().sort_values(ascending=False)
    displayed = list(importance.head(15).index[::-1])
    rng = np.random.default_rng(20261016)
    rng.choice(int(d.split.eq('holdout').sum()), size=len(sample), replace=False)
    for feature in displayed:
        jitter = rng.uniform(-.28, .28, len(sample))
        if feature == 'bathroom_count':
            break
    data = pd.DataFrame({'id': sample.id, 'bathroom_count': values,
                         'shap_value': saved.bathroom_count, 'vertical_jitter': jitter,
                         'normalized_color': normalized})
    data.to_csv(root / 'results/bathroom_shap_clue.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    f, ax = plt.subplots(figsize=(10, 2.7))
    sc = ax.scatter(saved.bathroom_count, jitter, c=normalized, cmap='coolwarm',
                    vmin=0, vmax=1, s=13, alpha=.5, edgecolors='none')
    ax.axvline(0, color='#888888', lw=1)
    ax.set(yticks=[0], yticklabels=['bathroom_count'], ylim=(-.55,.55),
           xlabel='TreeSHAP: вклад числа ванных в прогноз log(цены)',
           title='Зацепка: одна строка исходного SHAP-графика')
    ax.spines[['top','right','left']].set_visible(False)
    cb = f.colorbar(sc, ax=ax, pad=.025, fraction=.035, ticks=[0,1])
    cb.ax.set_yticklabels([f'≤{lo:g}', f'≥{hi:g}'])
    cb.set_label('Число ванных')
    f.text(.5,.015,'Те же 1 000 объявлений, SHAP-значения, цвета и вертикальное рассеивание.\n'
           'Положение по вертикали служит только для разделения точек; пропуски имеют нейтральный цвет.',
           ha='center',fontsize=9)
    f.tight_layout(rect=(0,.16,1,1))
    f.savefig(root/'figures/bathroom_shap_clue.png',dpi=160,bbox_inches='tight');plt.close(f)
    return data
