"""Generates notebooks/kaggle_matrix.ipynb: the two missing cells of the supervisor's experiment matrix."""
import json, os
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": s.strip("\n").splitlines(True)})
md("""
# Supervisor's matrix: the two missing cells

* **Weak VLM (LLaVA-1.5) + SLM compensator**: LLaVA-1.5 writes the same structured face description as InternVL3,
  then the SLM (Qwen2.5-1.5B + LoRA) is trained to keep / change / abstain on LLaVA's answer.
* **Strong VLM (InternVL3-8B) + SVLM compensator**: Qwen2.5-VL-3B + LoRA corrects InternVL3, same recipe as the
  Phase 1 SVLM arm for LLaVA.

**Inputs** (Add Input): **RAF-DB DATASET**, **Speech emotion VLM** (newest code), **notebooke3145f0d3e**,
**xfer-S2B** and **notebooke4d75ddfbf**.
**Settings**: Accelerator **GPU T4 x2**, **Internet on**. Then **Save Version -> Save & Run All** (about 2 to 2.5 hours;
you can close the tab). At the end download **matrix_results.zip** from the Output tab and send it to me.
""")
code("""
import os, glob, shutil, subprocess, sys, time
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '32_svlm_v2.py'))]
assert src, 'Old code found: upload the newest xfer-code.zip as a new version of Speech emotion VLM (then Check for updates)'
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
os.makedirs(f'{OUT}/predictions', exist_ok=True)
for s in ('train', 'val', 'test'):     # LLaVA-1.5's Phase 1 answers, shipped with the code
    shutil.copy2(f'{CODE}/bundle/V_{s}.csv', f'{OUT}/predictions/V_{s}.csv')
need = {'manifest.csv': 'notebooke3145f0d3e', 'results/D_geometry.csv': 'xfer-S2B',
        **{f'predictions/V2_{s}.csv': 'notebooke4d75ddfbf' for s in ('train', 'val', 'test')},
        'descriptions/V2_train_describe.csv': 'notebooke4d75ddfbf', 'descriptions/V2_train_given.csv': 'notebooke4d75ddfbf'}
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
IDS = f'{CODE}/bundle/V_train_ids.csv'
LLAVA = 'python scripts/18_run_vlm.py --out {o} --task describe --tag V --model llava-hf/llava-1.5-7b-hf'
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
        print(f'===== {name} (exit {p.returncode})'); print(open(f'/kaggle/working/log_{name}.txt').read()[-2500:])
        assert p.returncode == 0, name + ' failed: send me a screenshot of the log above'
""")
md("## Quick tests on a few images (stops here if anything is wrong)")
code("""
T = f'{OUT}/_smoke'
if os.path.exists(T): shutil.rmtree(T)
shutil.copytree(OUT, T)
tests = [('LLaVA describe', 'CUDA_VISIBLE_DEVICES=0 ' + LLAVA.format(o=T) + f' --splits train --ids_from {IDS} --limit 8'
                            ' && CUDA_VISIBLE_DEVICES=0 ' + LLAVA.format(o=T) + ' --splits val,test --limit 8'),
         ('SLM for LLaVA', f'CUDA_VISIBLE_DEVICES=0 python scripts/24_slm.py --out {T} --vlm V --limit 8 --epochs 1 --shap_n 1'),
         ('SVLM for InternVL3', f'CUDA_VISIBLE_DEVICES=1 python scripts/32_svlm_v2.py --out {T} --limit 6 --epochs 1')]
for name, cmd in tests:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'--- {name}: exit {r.returncode}'); print(r.stdout[-1500:])
    if r.returncode: print(r.stderr[-2500:])
    assert r.returncode == 0, name
print(pd.read_csv(f'{T}/descriptions/V_val_describe.csv').head(3).to_string())
shutil.rmtree(T)
""")
md("## Full runs")
code("""
parallel([
  ('gpu0', 'CUDA_VISIBLE_DEVICES=0 ' + LLAVA.format(o=OUT) + f' --splits train --ids_from {IDS}'
           ' && CUDA_VISIBLE_DEVICES=0 ' + LLAVA.format(o=OUT) + ' --splits val'),
  ('gpu1', 'CUDA_VISIBLE_DEVICES=1 ' + LLAVA.format(o=OUT) + ' --splits test'
           f' && CUDA_VISIBLE_DEVICES=1 python scripts/32_svlm_v2.py --out {OUT}'),
])
parallel([('slm', f'CUDA_VISIBLE_DEVICES=0 python scripts/24_slm.py --out {OUT} --vlm V --shap_n 20')])
""")
md("## Results and download")
code("""
for f in ['results/svlm_v2.md', 'results/slm_V.md']:
    print(open(f'{OUT}/{f}').read(), '\\n')
os.chdir('/kaggle/working')
subprocess.run('zip -qr matrix_results.zip outputs/predictions/SVLM2_* outputs/predictions/Vs_* '
               'outputs/descriptions/V_*_describe.csv outputs/slm_V/*.csv outputs/results/svlm_v2.md '
               'outputs/results/slm_V.md log_*.txt', shell=True, check=True)
print('download matrix_results.zip from the Output tab and send it to me')
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_matrix.ipynb"), "w"), indent=1)
print("wrote")
