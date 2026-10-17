"""Build the standalone theoretical notebook from its reviewed Russian source."""
from pathlib import Path
import re
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'docs/host-portfolio-economic-model.md').read_text()
parts=re.split(r'(?=^## \d+\.)',source,flags=re.MULTILINE)
cells=[]
for i,part in enumerate(parts):
    part=re.sub(r'!\[[^\]]*\]\([^\n]+\)\n?', '',part)
    part=part.replace('(../','(').strip()
    cells.append(nb.v4.new_markdown_cell(part,id=f'portfolio-theory-text-{i}'))
    if i==0:
        cells.append(nb.v4.new_code_cell('''from pathlib import Path
import pandas as pd
from IPython.display import display, Image
from helpers.host_portfolio_cost_model import build
theory_examples, theory_method = build()
def theory_figure(name):
    display(Image(filename=str(Path('figures') / f'host_portfolio_cost_model_{name}.png')))
''',id='portfolio-theory-compute',metadata={'jupyter':{'source_hidden':True}}))
    if i in [2,5,6]:
        name={2:'inputs',5:'solution',6:'scenarios'}[i]
        cells.append(nb.v4.new_code_cell(f"theory_figure('{name}')",id=f'portfolio-theory-plot-{name}',
                                       metadata={'jupyter':{'source_hidden':True}}))
cells.append(nb.v4.new_code_cell("display(pd.Series(theory_method['checks'], name='Проверка решения'))",
    id='portfolio-theory-checks',metadata={'jupyter':{'source_hidden':True}}))
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
    'language_info':{'name':'python','version':'3.12'}})
nb.write(n,ROOT/'host_portfolio_economic_model.ipynb')
print(len(cells),'cells built')
