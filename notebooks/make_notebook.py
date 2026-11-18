"""Generates notebooks/kaggle_E1.ipynb (run: python notebooks/make_notebook.py)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# Experiment E1: each source on its own (RAF-DB)

**Before running** (right-hand panel, *Session options*):
1. **Accelerator:** GPU T4 x2. **Internet:** On (needs a phone-verified Kaggle account).
2. **Add Input:** the RAF-DB dataset copy, and the `xfer-code` dataset (the project code uploaded as a zip).
3. Optional: if you are continuing from an earlier run, add that notebook's output as an input too.

Steps 1, 2 and 5 run on CPU in minutes. Steps 3 and 4 need the GPU and can take hours, so run them as
**Save Version → Save & Run All** and they continue after you close the browser. Every step resumes where it stopped.
""")

code("""
import os, glob, shutil, subprocess, sys
# 1) find the code and copy it to a writable folder
src = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)]
assert src, 'Add the xfer-code dataset as an input'
CODE = '/kaggle/working/xfer'
if not os.path.exists(CODE):
    shutil.copytree(os.path.dirname(src[0]), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt', shell=True, check=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv
""")

code("""
# 2) where the RAF-DB copy is, and where outputs go
cands = [d for d in glob.glob('/kaggle/input/*') if 'xfer' not in d.lower()]
print('inputs:', cands)
RAF_ROOT = next((d for d in cands if 'raf' in d.lower()), cands[0])   # change this line if the guess is wrong
OUT = '/kaggle/working/outputs'
# continuing from an earlier version? copy its outputs in first
prev = glob.glob('/kaggle/input/**/outputs/manifest.csv', recursive=True)
if prev and not os.path.exists(OUT):
    shutil.copytree(os.path.dirname(prev[0]), OUT)
    print('resumed outputs from', os.path.dirname(prev[0]))
print('RAF_ROOT =', RAF_ROOT)
""")

md("## Step 1. Data preparation (manifest, near-duplicates, train/val split, statistics)")
code("""
if not os.path.exists(f'{OUT}/manifest.csv'):
    !python scripts/01_prepare_data.py --root "$RAF_ROOT" --out "$OUT"
print(open(f'{OUT}/class_counts.md').read())
print(open(f'{OUT}/data_report.json').read())
""")
code("""
from IPython.display import Image, display
display(Image(f'{OUT}/figures/contact_sheet.png'))   # every row must show the expression named on its left
display(Image(f'{OUT}/figures/class_counts.png'))
""")

md("## Step 2. Source C: HOG/LBP + SVM (CPU)")
code("""
if not os.path.exists(f'{OUT}/predictions/C_test.csv'):
    !python scripts/02_run_svm.py --out "$OUT"
""")

md("## Step 3. Sources A (CLIP) and V (LLaVA-1.5), zero-shot. Smoke test first, then the full run")
code("""
!python scripts/03_run_zero_shot.py --out "$OUT" --source A --splits test --limit 32
!python scripts/03_run_zero_shot.py --out "$OUT" --source V --splits test --limit 16
""")
code("""
# full runs (resumable). Train-split predictions of V are needed later by the compensator.
!python scripts/03_run_zero_shot.py --out "$OUT" --source A --splits test,val,train
!python scripts/03_run_zero_shot.py --out "$OUT" --source V --splits test,val
""")

md("## Step 4. Source B: POSTER++ (train on train split, choose epoch on val, then predict)")
code("""
!python scripts/04_train_poster.py --out "$OUT" --work /kaggle/working/poster --epochs 60
""")

md("## Step 5. Evaluate E1")
code("""
!python scripts/05_evaluate_e1.py --out "$OUT" --split test
for f in sorted(glob.glob(f'{OUT}/results/confusion_*_test.png')):
    display(Image(f))
""")
code("""
# bundle the small outputs for download (predictions, results, figures; not model weights)
!cd /kaggle/working && zip -qr e1_outputs.zip outputs -x 'outputs/features/*' 'outputs/models/*'
print('download /kaggle/working/e1_outputs.zip from the Output panel')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_E1.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
