"""Generates notebooks/kaggle_stage2.ipynb (run: python notebooks/make_notebook_stage2.py)."""
import json
import os
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "A"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Stage 2: explanations, landmarks, calibration and the surrogate tree

This one notebook is used **twice**, as two Kaggle notebooks, because the work does not fit one 12-hour session:

| Notebook | `RUN` | Inputs to add | Does | Time |
|---|---|---|---|---|
| `xfer-S2A` | `'A'` | RAF-DB, `xfer-code` (new version), your E1 notebook | landmarks, neutral check, LLaVA on the training split, calibration, explanations for A, B, C, LLaVA explanations on **test** | about 9 h |
| `xfer-S2B` | `'B'` | the same three **plus** `xfer-S2A` | LLaVA explanations on **validation** and a 2,000-image **training** subset, surrogate tree, final tables | about 9.5 h |

Settings for both: **GPU T4 x2**, **Internet on**, then **Save Version → Save & Run All**.
""")

code(f"""
RUN = '{RUN}'          # 'A' for the first notebook (xfer-S2A), 'B' for the second (xfer-S2B)
""")

code("""
import os, glob, shutil, subprocess, sys, time
# code: prefer the uploaded xfer-code dataset over old copies inside earlier notebook outputs
# use the copy that has the stage-2 scripts (a dataset version may also hold the older zip)
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '09_region_explain.py'))]
assert src, 'No stage-2 code found: add the newest version of the xfer-code dataset as an input'
print('code from', src[0])
CODE = '/kaggle/working/xfer'
if not os.path.exists(CODE):
    shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
assert os.path.exists('scripts/09_region_explain.py'), 'xfer-code is the old version: upload the new zip as a new dataset version'
subprocess.run('pip install -q -r requirements-kaggle.txt', shell=True, check=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code("""
# data and earlier outputs: every input that holds an outputs/ folder is merged in (existing files are kept)
hits = [h for h in glob.glob('/kaggle/input/**/train_labels.csv', recursive=True) if 'xfer' not in h.lower()]
hits += glob.glob('/kaggle/input/**/list_patition_label.txt', recursive=True)
assert hits, 'Add the RAF-DB dataset as an input'
RAF_ROOT = os.path.dirname(hits[0])
OUT, WORK = '/kaggle/working/outputs', '/kaggle/working/poster'

def merge_tree(src, dst, skip=()):
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for f in files:
            if f in skip:
                continue
            d = os.path.join(dst, rel, f)
            if not os.path.exists(d):
                os.makedirs(os.path.dirname(d), exist_ok=True)
                shutil.copy2(os.path.join(root, f), d)

for man in glob.glob('/kaggle/input/**/outputs/manifest.csv', recursive=True):
    merge_tree(os.path.dirname(man), OUT); print('merged outputs from', os.path.dirname(man))
for rep in glob.glob('/kaggle/input/**/poster/POSTER_V2', recursive=True):
    merge_tree(os.path.dirname(rep), WORK, skip=('last.pth',)); print('merged POSTER++ files from', os.path.dirname(rep))
assert os.path.exists(f'{OUT}/predictions/V_test.csv'), 'Add your E1 notebook as an input'
print('RAF_ROOT =', RAF_ROOT)
""")

code("""
def parallel(cmds, every=600):
    '''Run shell commands at the same time, each with its own log; print the log tails every `every` seconds.'''
    procs = []
    for name, cmd in cmds:
        log = open(f'/kaggle/working/log_{name}.txt', 'a')
        procs.append((name, subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)))
    while any(p.poll() is None for _, p in procs):
        time.sleep(every)
        for name, p in procs:
            tail = open(f'/kaggle/working/log_{name}.txt').read().strip().splitlines()[-1:]
            print(f'[{name}]', 'running' if p.poll() is None else f'finished ({p.returncode})', '|', *tail)
    for name, p in procs:
        print(f'===== {name} (exit {p.returncode})'); print(open(f'/kaggle/working/log_{name}.txt').read()[-1500:])
        assert p.returncode == 0, f'{name} failed: see the log above'
""")

md("## Step 6. Landmarks, regions and geometry (source D), both runs, CPU, minutes")
code("""
if not os.path.exists(f'{OUT}/results/D_geometry.csv'):
    !python scripts/06_landmarks.py --out "$OUT"
print(open(f'{OUT}/results/D_by_class.csv').read())
from IPython.display import Image, display
display(Image(f'{OUT}/figures/region_masks.png'))   # the grey regions should sit on brows, eyes, nose, mouth...
""")

md("## Run A only: neutral check, LLaVA on the training split, calibration")
code("""
if RUN == 'A':
    if not os.path.exists(f'{OUT}/results/V_free_answers.csv'):
        !python scripts/07_neutral_check.py --out "$OUT"
    !python scripts/03_run_zero_shot.py --out "$OUT" --source V --splits train
    !python scripts/08_calibrate.py --out "$OUT" --sources V,A,B,C
    !python scripts/05_evaluate_e1.py --out "$OUT" --split test --sources V,Vcal,A,Acal,B,C --tag cal
""")

md("## Run A only: explanations for A, B (one GPU each) and C (CPU), all at the same time")
code("""
if RUN == 'A':
    S = 'test,val,train'
    parallel([
        ('A', f'CUDA_VISIBLE_DEVICES=0 python scripts/09_region_explain.py --out {OUT} --source A --splits {S} --batch 32'),
        ('B', f'CUDA_VISIBLE_DEVICES=1 python scripts/09_region_explain.py --out {OUT} --source B --splits {S} --batch 32 --work {WORK}'),
        ('C', f'python scripts/09_region_explain.py --out {OUT} --source C --splits {S}'),
    ], every=300)
    for s in 'ABC':
        !python scripts/09_region_explain.py --out "$OUT" --source {s} --splits test,val,train --merge
""")

md("## LLaVA explanations, split across the two GPUs. Run A: test. Run B: validation, then 1,500 training images (a subset of the 2,000 used for A, B, C)")
code("""
SPLITS = {'A': ['test'], 'B': ['val', 'train']}[RUN]
for split in SPLITS:
    parallel([(f'V{g}_{split}', f'CUDA_VISIBLE_DEVICES={g} python scripts/09_region_explain.py --out {OUT} '
               f'--source V --splits {split} --shard {g} --nshards 2 --batch 16 --train_subset 1500') for g in (0, 1)])
    !python scripts/09_region_explain.py --out "$OUT" --source V --splits {split} --merge
""")

md("## Run B only: surrogate tree and the stage-2 summary")
code("""
if RUN == 'B':
    !python scripts/10_surrogate_tree.py --out "$OUT"
    !python scripts/10_surrogate_tree.py --out "$OUT" --with_shap --choose 4
""")
code("""
import pandas as pd
for f in ['calibration.csv', 'cal_test.md', 'V_free_vs_scored.csv', 'surrogate_fidelity.csv']:
    p = f'{OUT}/results/{f}'
    if os.path.exists(p):
        print('=====', f); print(open(p).read())
for p in sorted(glob.glob(f'{OUT}/explanations/*_*.csv')):
    e = pd.read_csv(p)
    print(os.path.basename(p), len(e), 'images; median SHAP-LIME agreement', round(e.spearman.median(), 3))
!cd /kaggle/working && zip -qr stage2_{RUN}_results.zip outputs/results outputs/figures outputs/explanations/*.csv
print('small results bundle: /kaggle/working/stage2_' + RUN + '_results.zip')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_stage2.ipynb" if RUN == "A" else "kaggle_stage2B.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
