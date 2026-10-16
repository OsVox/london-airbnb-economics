"""Execute every cell in a new kernel using this exact interpreter."""
from pathlib import Path
import os,sys,json,tempfile
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='airbnb-kernel-') as tmp:
 k=Path(tmp)/'kernels/python3';k.mkdir(parents=True)
 (k/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Airbnb project','language':'python'}))
 os.environ['JUPYTER_PATH']=tmp
 os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4';os.environ['VECLIB_MAXIMUM_THREADS']='4'
 path=ROOT/'london_airbnb_group_project.ipynb';nb=nbformat.read(path,as_version=4)
 client=NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
 try:client.execute()
 finally:nbformat.write(nb,path)
 print('Executed',sum(c.cell_type=='code' for c in nb.cells),'code cells in a clean kernel.',flush=True)
