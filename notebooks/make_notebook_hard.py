"""Generates notebooks/kaggle_hard_images.ipynb: the image-level analysis of POSTER++'s test errors (CPU, minutes)."""
import json, os
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": s.strip("\n").splitlines(True)})
md("""
# P10: what is special about the test images POSTER++ gets wrong (brightness, contrast, sharpness, head turn)

**Inputs**: **RAF-DB DATASET**, **Speech emotion VLM**, **notebooke3145f0d3e**, **xfer-S2B**, **notebooke4d75ddfbf**.
**Settings**: no GPU needed (Accelerator **None**). Click **Run All**; it takes about 5 minutes. Then download
**hard_images_results.zip** from the Output panel (or the link printed at the end) and send it to me.
""")
code("""
import os, glob, shutil, subprocess, sys
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '25_hard_images.py'))]
assert src, 'Code not found: add Speech emotion VLM as an input'
CODE = '/kaggle/working/xfer'
os.chdir('/kaggle/working')
if os.path.exists(CODE): shutil.rmtree(CODE)
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
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
need = {'manifest.csv': 'notebooke3145f0d3e', 'predictions/B_test.csv': 'notebooke3145f0d3e',
        'predictions/V2_test.csv': 'notebooke4d75ddfbf', 'features/landmarks.npz': 'xfer-S2B'}
for f, who in need.items():
    assert os.path.exists(f'{OUT}/{f}'), f'{f} missing: add {who} as an input'
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
""")
code("""
r = subprocess.run(f'python scripts/25_hard_images.py --out {OUT}', shell=True)
assert r.returncode == 0, 'failed: send me a screenshot of the error above'
os.chdir('/kaggle/working')
subprocess.run('zip -qj hard_images_results.zip outputs/results/hard_images.md outputs/results/hard_images_test.csv '
               'outputs/figures/poster_errors_sheet_*.png', shell=True, check=True)
shutil.rmtree(OUT); shutil.rmtree(CODE)
from IPython.display import FileLink
FileLink('hard_images_results.zip')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_hard_images.ipynb"), "w"), indent=1)
print("wrote")
