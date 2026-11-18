"""Generates notebooks/kaggle_step2_occlusion.ipynb (run: python notebooks/make_notebook_step2.py)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# New design, step 2: the shared explanation (region occlusion) for all four experts

**Inputs** (Add Input): **RAF-DB DATASET**, **Speech emotion VLM** (newest version of the code),
the notebook **notebooke3145f0d3e** (trained POSTER++ and SVM), the notebook **xfer-S2B** (face landmarks),
and the step-1 notebook **notebooke4d75ddfbf**.
**Settings**: Accelerator **GPU T4 x2**, **Internet on**. Then **Save Version → Save & Run All**
(background run, about 1.5 to 2 hours; if it stops, run it again and it continues where it left off).

For every image, each of 10 face regions is blurred in turn and the drop in each emotion's probability is recorded;
then deletion and insertion curves test whether the ranking is faithful. CLIP, POSTER++ and the SVM take minutes;
the new VLM (InternVL3-8B) explains a class-stratified sample of 800 test and 500 validation images, split over
both GPUs.
""")

code("""
import os, glob, shutil, subprocess, sys, time
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '20_occlusion.py'))]
assert src, 'No step-2 code found: upload the newest xfer-code.zip as a new version of your code dataset'
CODE = '/kaggle/working/xfer'
if not os.path.exists(CODE):
    shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt', shell=True, check=True)
import importlib.metadata as md
from packaging.version import Version
if Version(md.version('transformers')) < Version('4.52'):
    subprocess.run('pip install -q -U "transformers>=4.52"', shell=True, check=True)
for pkg in ['torch', 'transformers', 'bitsandbytes', 'timm']:
    print(pkg, md.version(pkg))
""")

md("## 1. Collect the splits, landmarks and trained models")
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
if not os.path.exists(m.path.iloc[0]):
    idx = {}
    for dp, _, fs in os.walk('/kaggle/input'):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx.setdefault(f, os.path.join(dp, f))
    m['path'] = m.path.map(lambda p: idx.get(os.path.basename(p), p))
    m.to_csv(f'{OUT}/manifest.csv', index=False)
print('images found:', m.path.map(os.path.exists).mean())
assert os.path.exists(f'{OUT}/features/landmarks.npz'), 'landmarks missing: add xfer-S2B as an input'
assert os.path.exists(f'{OUT}/models/svm_C.joblib'), 'SVM missing: add notebooke3145f0d3e as an input'
SUBSET = f'{OUT}/explanations/V_train.csv'
assert os.path.exists(SUBSET), 'train subset missing: add xfer-S2B as an input'
ck = glob.glob('/kaggle/input/**/checkpoints_seed42/best.pth', recursive=True)
assert ck, 'POSTER++ checkpoint missing: add notebooke3145f0d3e as an input'
CKPT, WORK = ck[0], '/kaggle/working/poster'
for sub in ('POSTER_V2', 'pretrain'):
    s = os.path.join(os.path.dirname(os.path.dirname(CKPT)), sub)
    if os.path.isdir(s) and not os.path.exists(os.path.join(WORK, sub)):
        shutil.copytree(s, os.path.join(WORK, sub))
print('POSTER++:', CKPT)
""")

code("""
def parallel(cmds, every=900):
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
        print(f'===== {name} (exit {p.returncode})'); print(open(f'/kaggle/working/log_{name}.txt').read()[-2500:])
        assert p.returncode == 0, f'{name} failed: see the log above'
OCC = f'python scripts/20_occlusion.py --out {OUT} --work {WORK} --ckpt {CKPT}'
""")

md("## 2. Quick test: 3 validation images per expert (stops here if anything is wrong)")
code("""
T = f'{OUT}/_smoke'
if os.path.exists(T): shutil.rmtree(T)
for sub in ('features', 'models'):
    shutil.copytree(f'{OUT}/{sub}', f'{T}/{sub}')
shutil.copy(f'{OUT}/manifest.csv', T)
for src_ in ['C', 'A', 'B', 'V2']:
    r = subprocess.run(f'CUDA_VISIBLE_DEVICES=0 python scripts/20_occlusion.py --out {T} --work {WORK} --ckpt {CKPT} '
                       f'--source {src_} --splits val --limit 3 --chunk 3 && '
                       f'python scripts/20_occlusion.py --out {T} --source {src_} --splits val --merge',
                       shell=True, capture_output=True, text=True)
    print(src_, r.stdout[-700:], r.stderr[-1200:] if r.returncode else ''); assert r.returncode == 0, src_
print(pd.read_csv(f'{T}/explanations/occ_V2_val.csv').round(3).T)
shutil.rmtree(T)
""")

md("## 3. Full runs: the three fast experts on every image, then the VLM sample on both GPUs")
code("""
# fast mode: the VLM explains a class-stratified sample (800 test, 500 validation images) with a 1-step
# faithfulness test (13 calls per image); CLIP, POSTER++ and the SVM explain every image
V2 = lambda g: (f'CUDA_VISIBLE_DEVICES={g} {OCC} --source V2 --splits test --sample 800 --k 1 --shard {g} --nshards 2'
                f' && CUDA_VISIBLE_DEVICES={g} {OCC} --source V2 --splits val --sample 500 --k 1 --shard {g} --nshards 2')
parallel([
  ('gpu0', f'CUDA_VISIBLE_DEVICES=0 {OCC} --source A --splits val,test && ' + V2(0)),
  ('gpu1', f'CUDA_VISIBLE_DEVICES=1 {OCC} --source B --splits val,test && CUDA_VISIBLE_DEVICES=1 {OCC} --source C '
           f'--splits val,test && ' + V2(1)),
])
""")

md("## 4. Merge and summarise")
code("""
for s_, splits in [('A', 'val,test'), ('B', 'val,test'), ('C', 'val,test'), ('V2', 'val,test')]:
    subprocess.run(f'python scripts/20_occlusion.py --out {OUT} --source {s_} --splits {splits} --merge', shell=True)
rows = []
for s_ in ['V2', 'A', 'B', 'C']:
    f = f'{OUT}/explanations/occ_{s_}_test.csv'
    if os.path.exists(f):
        d = pd.read_csv(f)
        rows.append(dict(source=s_, n=len(d), deletion_area=round(d.deletion_area.mean(), 3),
                         insertion_area=round(d.insertion_area.mean(), 3),
                         top_regions=d.top_region.value_counts(normalize=True).head(3).round(2).to_dict()))
summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
os.makedirs(f'{OUT}/results', exist_ok=True)
summary.to_csv(f'{OUT}/results/occlusion_summary_test.csv', index=False)
!cd /kaggle/working && zip -qr step2_results.zip outputs/explanations/occ_*.csv outputs/explanations/occ_*.npz outputs/results/occlusion_summary_test.csv
print('results bundle: /kaggle/working/step2_results.zip (download it and send it to me)')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_step2_occlusion.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
