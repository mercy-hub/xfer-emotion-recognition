"""Generates notebooks/kaggle_demo.ipynb (run: python notebooks/make_notebook_demo.py)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Demo app: explainable facial emotion recognition

**Inputs** (Add Input): **RAF-DB DATASET**, **Speech emotion VLM** (newest version of the code),
the E1 notebook **notebooke3145f0d3e** (trained POSTER++ and SVM), and the notebook **xfer-S2B**.
**Settings**: Accelerator **GPU T4 x2** (one GPU is used), **Internet on**.

**Run it in the editor with Run All** (not Save Version), because the app lives only while this session is open.
After about 10 minutes the last cell prints a public link ending in **gradio.live**. Open it on any phone or laptop.
Keep this browser tab open during the demonstration; the link works for up to 72 hours while the session runs.
""")

code("""
import os, glob, shutil, subprocess, sys
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '17_demo_app.py'))]
assert src, 'No demo code found: upload the newest xfer-code.zip as a new version of your code dataset'
CODE = '/kaggle/working/xfer'
if not os.path.exists(CODE):
    shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt "gradio>=5,<6"', shell=True, check=True)
print('code from', src[0])
""")

md("## 1. Collect the stored results and the trained models")
code("""
OUT = '/kaggle/working/outputs'
def merge_tree(s, d):
    for root, _, files in os.walk(s):
        for f in files:
            t = os.path.join(d, os.path.relpath(root, s), f)
            if not os.path.exists(t):
                os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(os.path.join(root, f), t)
for man in glob.glob('/kaggle/input/**/outputs/manifest.csv', recursive=True):
    merge_tree(os.path.dirname(man), OUT); print('merged', os.path.dirname(man))
need = {'explanations/V_train.csv': 'xfer-S2B', 'results/D_geometry.csv': 'xfer-S2B',
        'predictions/VT_val.csv': 'xfer-S2B'}
for f, who in need.items():
    print(('OK      ' if os.path.exists(f'{OUT}/{f}') else 'MISSING ') + f + ('' if os.path.exists(f'{OUT}/{f}') else '  -> add ' + who))

# image paths may have moved since E1: point the manifest at the files under /kaggle/input
import pandas as pd
m = pd.read_csv(f'{OUT}/manifest.csv')
if not os.path.exists(m.path.iloc[0]):
    idx = {}
    for dp, _, fs in os.walk('/kaggle/input'):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx.setdefault(f, os.path.join(dp, f))
    m['path'] = m.path.map(lambda p: idx.get(os.path.basename(p), p))
    m.to_csv(f'{OUT}/manifest.csv', index=False)
print('images found:', m.path.map(os.path.exists).mean())
if not os.path.exists(f'{OUT}/results/traces_test.csv'):     # stored decisions for the examples tab (CPU, minutes)
    subprocess.run(f'python scripts/11_reasoner.py --out {OUT}', shell=True, check=True)
""")

code("""
# POSTER++ checkpoint and code (from the E1 notebook output); retrain only if it is missing
ck = glob.glob('/kaggle/input/**/checkpoints_seed42/best.pth', recursive=True)
WORK = '/kaggle/working/poster'
if ck:
    CKPT = ck[0]
    srcwork = os.path.dirname(os.path.dirname(CKPT))
    for sub in ('POSTER_V2', 'pretrain'):
        if os.path.isdir(os.path.join(srcwork, sub)) and not os.path.exists(os.path.join(WORK, sub)):
            shutil.copytree(os.path.join(srcwork, sub), os.path.join(WORK, sub))
    print('POSTER++ checkpoint:', CKPT)
else:
    print('No POSTER++ checkpoint found in the inputs: training it now (about 3 hours).')
    subprocess.run(f'python scripts/04_train_poster.py --out {OUT} --work {WORK} --epochs 60', shell=True, check=True)
    CKPT = f'{WORK}/checkpoints_seed42/best.pth'
if not os.path.exists(f'{OUT}/models/svm_C.joblib'):
    print('No SVM model found: training it now (a few minutes).')
    subprocess.run(f'python scripts/02_run_svm.py --out {OUT}', shell=True, check=True)
print('SVM:', f'{OUT}/models/svm_C.joblib')
""")

md("## 2. Start the app (models load in about 10 minutes the first time). The public link (…gradio.live) appears below; this cell keeps running while the app is open.")
code("""
!CUDA_VISIBLE_DEVICES=0 python scripts/17_demo_app.py --out {OUT} --ckpt {CKPT} --work {WORK} --share
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_demo.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
