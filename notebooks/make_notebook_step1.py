"""Generates notebooks/kaggle_step1_vlm.ipynb (run: python notebooks/make_notebook_step1.py)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# New design, step 1: the new VLM (InternVL3-8B)

**Inputs** (Add Input): **RAF-DB DATASET**, **Speech emotion VLM** (newest version of the code),
the notebook **notebooke3145f0d3e** (splits and the old LLaVA results), and the notebook **xfer-S2B**
(the 1,501 explained training images).
**Settings**: Accelerator **GPU T4 x2**, **Internet on**. Then **Save Version → Save & Run All**
(it runs in the background, about 4 to 6 hours; you can close the browser).

What it does: a quick test on 8 images, then two jobs at the same time:
GPU 0: class probabilities on val, test and train, then the *given-emotion* descriptions (train subset and val only).
GPU 1: the structured descriptions (emotion not given) on val, test and the train subset.
Last cell: accuracy of the new VLM against LLaVA-1.5, and a zip of the results.
""")

code("""
import os, glob, shutil, subprocess, sys, time
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '18_run_vlm.py'))]
assert src, 'No step-1 code found: upload the newest xfer-code.zip as a new version of your code dataset'
CODE = '/kaggle/working/xfer'
if not os.path.exists(CODE):
    shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt', shell=True, check=True)
import importlib.metadata as md
from packaging.version import Version
if Version(md.version('transformers')) < Version('4.52'):   # InternVL3-hf needs 4.52 or newer
    subprocess.run('pip install -q -U "transformers>=4.52"', shell=True, check=True)
for pkg in ['torch', 'transformers', 'bitsandbytes', 'accelerate']:
    print(pkg, md.version(pkg))
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

md("## 1. Collect the splits and earlier results")
code("""
import pandas as pd
OUT = '/kaggle/working/outputs'
def merge_tree(s, d):
    for root, _, files in os.walk(s):
        for f in files:
            t = os.path.join(d, os.path.relpath(root, s), f)
            if not os.path.exists(t):
                os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(os.path.join(root, f), t)
for man in glob.glob('/kaggle/input/**/outputs/manifest.csv', recursive=True):
    merge_tree(os.path.dirname(man), OUT); print('merged', os.path.dirname(man))
m = pd.read_csv(f'{OUT}/manifest.csv')
if not os.path.exists(m.path.iloc[0]):          # image paths moved since E1: point them at /kaggle/input
    idx = {}
    for dp, _, fs in os.walk('/kaggle/input'):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx.setdefault(f, os.path.join(dp, f))
    m['path'] = m.path.map(lambda p: idx.get(os.path.basename(p), p))
    m.to_csv(f'{OUT}/manifest.csv', index=False)
print('images found:', m.path.map(os.path.exists).mean(), '| per split:', m.split.value_counts().to_dict())
SUBSET = f'{OUT}/explanations/V_train.csv'
assert os.path.exists(SUBSET), 'Add the notebook xfer-S2B as an input (it holds the 1,501 explained train images)'
print('train subset:', len(pd.read_csv(SUBSET, usecols=['image_id'])), 'images')
""")

code("""
def parallel(cmds, every=600):
    procs = []
    for name, cmd in cmds:
        log = open(f'/kaggle/working/log_{name}.txt', 'a')
        procs.append((name, subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)))
    while any(p.poll() is None for _, p in procs):
        time.sleep(every)
        for name, p in procs:
            tail = open(f'/kaggle/working/log_{name}.txt').read().strip().splitlines()[-1:]
            print(f'[{name}]', 'running' if p.poll() is None else f'finished ({p.returncode})', '|', *tail, flush=True)
    for name, p in procs:
        print(f'===== {name} (exit {p.returncode})'); print(open(f'/kaggle/working/log_{name}.txt').read()[-3000:])
        assert p.returncode == 0, f'{name} failed: see the log above'
""")

md("## 2. Download the model once, then a quick test on 8 validation images (stops here if anything is wrong)")
code("""
from huggingface_hub import snapshot_download
snapshot_download('OpenGVLab/InternVL3-8B-hf')
T = f'{OUT}/_smoke'
if os.path.exists(T): shutil.rmtree(T)
os.makedirs(T); shutil.copy(f'{OUT}/manifest.csv', T)
for task in ['proba', 'describe', 'given']:
    r = subprocess.run(f'CUDA_VISIBLE_DEVICES=0 python scripts/18_run_vlm.py --out {T} --task {task} --splits val --limit 8',
                       shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:]); assert r.returncode == 0, task
print(pd.read_csv(f'{T}/predictions/V2_val.csv').round(3).to_string())
for t in ['describe', 'given']:
    d = pd.read_csv(f'{T}/descriptions/V2_val_{t}.csv')
    print(f'--- {t}'); [print(f'[{r.label}]', r.text) for r in d.head(4).itertuples()]
shutil.rmtree(T)
""")

md("## 3. Full runs on both GPUs (resumable: if the session stops, run the notebook again)")
code("""
S = f'--ids_from {SUBSET}'
parallel([
  ('gpu0', f'CUDA_VISIBLE_DEVICES=0 python scripts/18_run_vlm.py --out {OUT} --task proba --splits val,test,train'
           f' && CUDA_VISIBLE_DEVICES=0 python scripts/18_run_vlm.py --out {OUT} --task given --splits val'
           f' && CUDA_VISIBLE_DEVICES=0 python scripts/18_run_vlm.py --out {OUT} --task given --splits train {S}'),
  ('gpu1', f'CUDA_VISIBLE_DEVICES=1 python scripts/18_run_vlm.py --out {OUT} --task describe --splits val,test'
           f' && CUDA_VISIBLE_DEVICES=1 python scripts/18_run_vlm.py --out {OUT} --task describe --splits train {S}'),
])
""")

md("## 4. How good is the new VLM? (compared with LLaVA-1.5)")
code("""
!python scripts/19_eval_vlm.py --out "$OUT" --split test
!python scripts/19_eval_vlm.py --out "$OUT" --split val
!cd /kaggle/working && zip -qr step1_results.zip outputs/results/vlm_compare_* outputs/predictions/V2_* outputs/descriptions
print('results bundle: /kaggle/working/step1_results.zip (download it and send it to me)')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_step1_vlm.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
