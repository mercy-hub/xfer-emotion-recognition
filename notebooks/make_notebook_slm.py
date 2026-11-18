"""Generates notebooks/kaggle_slm_only.ipynb: retrains only the SLM compensator so its adapter is kept as output."""
import json, os
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": s.strip("\n").splitlines(True)})
md("""
# SLM only: retrain the SLM compensator and keep its adapter (slm/lora) for the demo app

**Inputs** (Add Input), five in total: **RAF-DB DATASET**, **Speech emotion VLM** (newest code),
**notebooke3145f0d3e**, **xfer-S2B** and **notebooke4d75ddfbf**.
**Settings**: Accelerator **GPU T4 x2**, **Internet on**. Then **Save Version -> Save & Run All** (not Run All:
only a saved version keeps its output). About 25 minutes.
""")
code("""
import os, glob, shutil, subprocess, sys
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '24_slm.py'))]
assert src, 'Code not found: add Speech emotion VLM as an input'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
CODE = '/kaggle/working/xfer'
os.chdir('/kaggle/working')
if os.path.exists(CODE): shutil.rmtree(CODE)
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt', shell=True, check=True)
subprocess.run('pip uninstall -y -q torchao', shell=True)
import importlib.metadata as md
from packaging.version import Version
if Version(md.version('transformers')) < Version('4.52'):
    subprocess.run('pip install -q -U "transformers>=4.52"', shell=True, check=True)
""")
code("""
import pandas as pd
OUT = '/kaggle/working/outputs'
def merge_tree(s, d):
    for root, _, files in os.walk(s):
        for f in files:
            t = os.path.join(d, os.path.relpath(root, s), f)
            if not os.path.exists(t):
                os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(os.path.join(root, f), t)
for d in sorted(glob.glob('/kaggle/input/**/outputs', recursive=True)):
    if os.path.isdir(d) and not d.startswith(CODE):
        merge_tree(d, OUT); print('merged', d)
need = {'manifest.csv': 'notebooke3145f0d3e', 'results/D_geometry.csv': 'xfer-S2B',
        **{f'predictions/V2_{s}.csv': 'notebooke4d75ddfbf' for s in ('train', 'val', 'test')},
        **{f'descriptions/V2_{s}_describe.csv': 'notebooke4d75ddfbf' for s in ('train', 'val', 'test')},
        'descriptions/V2_train_given.csv': 'notebooke4d75ddfbf'}
for f, who in need.items():
    assert os.path.exists(f'{OUT}/{f}'), f'{f} missing: add {who} as an input'
shutil.rmtree(f'{OUT}/slm', ignore_errors=True)       # train afresh
print('all inputs present')
""")
code("""
r = subprocess.run(f'CUDA_VISIBLE_DEVICES=0 python scripts/24_slm.py --out {OUT} --shap_n 20', shell=True)
assert r.returncode == 0, 'SLM training failed: send me a screenshot of the error above'
assert os.path.exists(f'{OUT}/slm/lora/adapter_config.json')
print(open(f'{OUT}/results/slm.md').read())
for d in os.listdir(OUT):                              # keep the output small: only the SLM folder is needed
    if d != 'slm':
        p = os.path.join(OUT, d); shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
shutil.rmtree(CODE)
print('done: the adapter is in outputs/slm/lora. Add this notebook as an input to Demo app 2.')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_slm_only.ipynb"), "w"), indent=1)
print("wrote")
