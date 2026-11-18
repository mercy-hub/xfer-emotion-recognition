"""Generates notebooks/kaggle_step4_complete.ipynb (run: python notebooks/make_notebook_step4.py)."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# New design, step 4: the remaining boxes of the supervisor's diagram

**Inputs** (Add Input): **RAF-DB DATASET**, **Speech emotion VLM** (newest version of the code),
the notebooks **notebooke3145f0d3e** (POSTER++ and SVM), **xfer-S2B** (landmarks), **notebooke4d75ddfbf** (step 1)
and **notebook51b7dde403** (step 2). If Kaggle fails to mount notebook51b7dde403, upload **step2_results.zip** as a
new dataset and add that instead.
**Settings**: Accelerator **GPU T4 x2**, **Internet on**. Then **Save Version → Save & Run All**
(background run, about 2 to 3 hours).

Runs at the same time:
* GPU 0: **UniFER-7B** (the emotion-trained VLM, as an extra expert) on validation and test, then the VLM's
  **answer-token relevance** on 300 test faces;
* GPU 1: the **SLM compensator** (Qwen2.5-1.5B + LoRA: keep / change / abstain + explanation, with field-level
  Shapley values), then **CLIP relevance maps** and **AU-phrase probing**, then **POSTER++ cross-attention relevance and
  Integrated Gradients on the landmark stream**;
* CPU: the **linear SVM** with exact weights.
""")

code("""
import os, glob, shutil, subprocess, sys, time
def _slm(p):
    f = os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '24_slm.py')
    return os.path.exists(f) and '--micro' in open(f).read()
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True) if _slm(p)]
assert src, 'Old code found: upload the newest xfer-code.zip as a new version of your code dataset (Speech emotion VLM)'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   # less GPU memory fragmentation
CODE = '/kaggle/working/xfer'
if not os.path.exists(CODE):
    shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt', shell=True, check=True)
# Kaggle's old torchao (0.10) makes the new peft refuse to add LoRA layers; we do not use torchao, so remove it
subprocess.run('pip uninstall -y -q torchao', shell=True)
import importlib.metadata as md
from packaging.version import Version
if Version(md.version('transformers')) < Version('4.52'):
    subprocess.run('pip install -q -U "transformers>=4.52"', shell=True, check=True)
for pkg in ['torch', 'transformers', 'bitsandbytes', 'peft', 'timm']:  # torchao should be gone
    print(pkg, md.version(pkg))
""")

md("## 1. Collect everything from the earlier runs")
code("""
import pandas as pd
OUT = '/kaggle/working/outputs'
def merge_tree(s, d):
    for root, _, files in os.walk(s):
        for f in files:
            t = os.path.join(d, os.path.relpath(root, s), f)
            if not os.path.exists(t):
                os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(os.path.join(root, f), t)
# every 'outputs' folder among the inputs (notebook outputs, or an uploaded results zip such as step2_results.zip)
for d in sorted(glob.glob('/kaggle/input/**/outputs', recursive=True)):
    if os.path.isdir(d) and not d.startswith(CODE):
        merge_tree(d, OUT); print('merged', d)
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
need = {'features/landmarks.npz': 'xfer-S2B', 'results/D_geometry.csv': 'xfer-S2B', 'predictions/V2_test.csv': 'notebooke4d75ddfbf',
        'descriptions/V2_train_given.csv': 'notebooke4d75ddfbf', 'explanations/occ_V2_test.npz': 'notebook51b7dde403 or the uploaded step2_results.zip dataset',
        'explanations/occ_B_test.npz': 'notebook51b7dde403 or the uploaded step2_results.zip dataset', 'models/svm_C.joblib': 'notebooke3145f0d3e'}
for f, who in need.items():
    assert os.path.exists(f'{OUT}/{f}'), f'{f} missing: add {who} as an input'
ck = glob.glob('/kaggle/input/**/checkpoints_seed42/best.pth', recursive=True)
assert ck, 'POSTER++ checkpoint missing: add notebooke3145f0d3e as an input'
CKPT, WORK = ck[0], '/kaggle/working/poster'
for sub in ('POSTER_V2', 'pretrain'):
    s = os.path.join(os.path.dirname(os.path.dirname(CKPT)), sub)
    if os.path.isdir(s) and not os.path.exists(os.path.join(WORK, sub)):
        shutil.copytree(s, os.path.join(WORK, sub))
open(f'{OUT}/results/native_xai.md', 'w').write('# Native explanations of each expert\\n\\n')
print('all inputs present; POSTER++:', CKPT)
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
    bad = []
    for name, p in procs:
        print(f'===== {name} (exit {p.returncode})'); print(open(f'/kaggle/working/log_{name}.txt').read()[-3000:])
        if p.returncode != 0: bad.append(name)
    print('FAILED:', bad if bad else 'none')
UNIFER = ('python scripts/18_run_vlm.py --out {o} --task proba --tag U --model Karl28/UniFER-7B --forms '
          '--answer_prefix "<think>\\\\n</think>\\\\n<answer>"')
NX = f'python scripts/23_native_xai.py --out {OUT} --work {WORK} --ckpt {CKPT}'
""")

md("## 2. Quick tests on a few images (stops here if anything is wrong)")
code("""
T = f'{OUT}/_smoke'
if os.path.exists(T): shutil.rmtree(T)
shutil.copytree(OUT, T)
tests = [('UniFER', 'CUDA_VISIBLE_DEVICES=0 ' + UNIFER.format(o=T) + ' --splits val --limit 8'),
         ('SLM', f'CUDA_VISIBLE_DEVICES=1 python scripts/24_slm.py --out {T} --limit 8 --epochs 1 --shap_n 1'),
         ('CLIP relevance', f'CUDA_VISIBLE_DEVICES=1 python scripts/23_native_xai.py --out {T} --method clip_rel --n 2'),
         ('POSTER++', f'CUDA_VISIBLE_DEVICES=1 python scripts/23_native_xai.py --out {T} --work {WORK} --ckpt {CKPT} --method poster --n 2 --ig_steps 2'),
         ('VLM relevance', f'CUDA_VISIBLE_DEVICES=0 python scripts/23_native_xai.py --out {T} --method vlm_rel --n 2')]
for name, cmd in tests:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'--- {name}: exit {r.returncode}'); print(r.stdout[-1500:])
    if r.returncode: print(r.stderr[-2500:])
    assert r.returncode == 0, name
print(pd.read_csv(f'{T}/predictions/U_val.csv').round(3).to_string())
shutil.rmtree(T)
""")

md("## 3. Full runs (both GPUs and the CPU at the same time)")
code("""
parallel([
  ('gpu0', 'CUDA_VISIBLE_DEVICES=0 ' + UNIFER.format(o=OUT) + ' --splits val,test'
           f' ; CUDA_VISIBLE_DEVICES=0 {NX} --method vlm_rel --n 300'),
  ('gpu1', f'CUDA_VISIBLE_DEVICES=1 python scripts/24_slm.py --out {OUT}'
           f' ; CUDA_VISIBLE_DEVICES=1 {NX} --method clip_rel --n 300'
           f' ; CUDA_VISIBLE_DEVICES=1 {NX} --method clip_au'
           f' ; CUDA_VISIBLE_DEVICES=1 {NX} --method poster --n 300'),
  ('cpu', f'CUDA_VISIBLE_DEVICES= {NX} --method linsvm'),
])
""")

md("## 4. Results and download")
code("""
!python scripts/19_eval_vlm.py --out "$OUT" --split test --tag U
for f in ['results/slm.md', 'results/native_xai.md']:
    p = f'{OUT}/{f}'
    print(open(p).read() if os.path.exists(p) else f'{f} missing')
!cd /kaggle/working && zip -qr step4_results.zip outputs/predictions/U_* outputs/predictions/V2s_* outputs/predictions/CL_* outputs/slm/slm_*.csv outputs/slm/shapley_test.csv outputs/explanations/native_* outputs/results/slm.md outputs/results/native_xai.md outputs/results/vlm_compare_test.md
print('results bundle: /kaggle/working/step4_results.zip (download it and send it to me)')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_step4_complete.ipynb")
json.dump(nb, open(path, "w"), indent=1)
print("wrote", path)
