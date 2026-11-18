"""Generates notebooks/kaggle_E4.ipynb (run: python notebooks/make_notebook_stage4.py)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Experiment E4: the post-hoc compensator

Inputs: **RAF-DB DATASET**, **Speech emotion VLM** (newest version of the code), **xfer-S2B** (your stage-2 run B).
Settings: **GPU T4 x2**, **Internet on**, then **Save Version → Save & Run All**. About 2 to 3 hours.

Steps: the CPU arm (detector and tabular corrector, minutes); a quick test of the two image arms; then the CNN arm on
GPU 0 and the Qwen2.5-VL arm on GPU 1 at the same time; then one table for all arms.
""")

code("""
import os, glob, shutil, subprocess, sys, time
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '16_full_framework.py'))]
assert src, 'No E4 code found: upload the newest xfer-code.zip as a new version of your code dataset'
print('code from', src[0])
CODE = '/kaggle/working/xfer'
if not os.path.exists(CODE):
    shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt', shell=True, check=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
import importlib.metadata as md, platform
print('python', platform.python_version())
for pkg in ['torch','transformers','bitsandbytes','accelerate','peft','timm','mediapipe','scikit-learn','numpy','pandas','owlready2']:
    try: print(pkg, md.version(pkg))
    except Exception: print(pkg, 'not installed')
""")

code("""
OUT = '/kaggle/working/outputs'
def merge_tree(src, dst):
    for root, _, files in os.walk(src):
        for f in files:
            d = os.path.join(dst, os.path.relpath(root, src), f)
            if not os.path.exists(d):
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(os.path.join(root, f), d)
for man in glob.glob('/kaggle/input/**/outputs/manifest.csv', recursive=True):
    merge_tree(os.path.dirname(man), OUT); print('merged outputs from', os.path.dirname(man))
assert os.path.exists(f'{OUT}/explanations/V_train.csv'), 'Add xfer-S2B as an input'
assert os.path.exists(f'{OUT}/features/landmarks.npz'), 'landmarks missing: add xfer-S2B as an input'
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

md("## 1. CPU arm: detector and tabular corrector (minutes)")
code("""
!python scripts/12_compensator_tabular.py --out "$OUT"
""")

md("## 2. Quick test of both image arms on a few images (a few minutes); stops here if something is wrong")
code("""
parallel([
  ('cnn_test', f'CUDA_VISIBLE_DEVICES=0 python scripts/13_compensator_vision.py --out {OUT} --epochs 1 --limit 16 --ablations image+pred+attr'),
  ('vlm_test', f'CUDA_VISIBLE_DEVICES=1 python scripts/14_compensator_vlm.py --out {OUT} --epochs 1 --limit 8 --ablations image+pred+expl'),
], every=60)
""")

md("## 3. Full runs: CNN arm on GPU 0, Qwen2.5-VL arm on GPU 1")
code("""
parallel([
  ('cnn', f'CUDA_VISIBLE_DEVICES=0 python scripts/13_compensator_vision.py --out {OUT}'),
  ('vlm', f'CUDA_VISIBLE_DEVICES=1 python scripts/14_compensator_vlm.py --out {OUT}'),
])
""")

md("## 4. One table for every arm (Table 18), then the full framework with each image arm's full input")
code("""
!python scripts/15_evaluate_e4.py --out "$OUT"
!python scripts/16_full_framework.py --out "$OUT" --tag Vcomp-pred+expl+face
import os
for t in ['Vcomp-cnn-image+pred+attr', 'Vcomp-vlm-image+pred+expl']:
    if os.path.exists(f'{OUT}/predictions/{t}_test.csv'):
        os.system(f'python scripts/16_full_framework.py --out "{OUT}" --tag {t} && cp "{OUT}/results/e2_full_test.md" "{OUT}/results/e2_full_{t}.md"')
!cd /kaggle/working && zip -qr e4_results.zip outputs/results outputs/predictions/det-* outputs/predictions/corr-* outputs/predictions/Vcomp-*
print('results bundle: /kaggle/working/e4_results.zip')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_E4.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
