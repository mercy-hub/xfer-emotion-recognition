"""Generates notebooks/kaggle_demo2.ipynb (run: python notebooks/make_notebook_demo2.py)."""
import json
import os

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": s.strip("\n").splitlines(True)})

md("""
# Demo app, Phase 2: the final system (four experts, XAI fusion, SLM compensator, EMONTO + FACS)

**Inputs** (Add Input), six in total:
* **RAF-DB DATASET** and **Speech emotion VLM** (newest version of the code);
* **notebooke3145f0d3e** (POSTER++, the SVM and the data list);
* the **step 4** notebook that finished (its output holds the SLM in `slm/lora`);
* **notebooka8c6aeaeff** (step 5: the fine-tuned CLIP, `experts/A2_best.pth`);
* nothing else is needed: InternVL3-8B, CLIP and Qwen2.5 download from Hugging Face.

**Settings**: Accelerator **GPU T4 x2**, **Internet on**. Run it in the **editor with Run All** (not Save Version):
the app lives only while this session is open. After about 10 to 15 minutes the last cell prints a public link
ending in **gradio.live**; open it on any phone or laptop and keep this tab open during the demonstration.
""")
code("""
import os, glob, shutil, subprocess, sys
def _new(p):   # the newest code: the demo can run without the SLM adapter
    f = os.path.join(os.path.dirname(p), 'demo2.py')
    return os.path.exists(f) and 'self.off' in open(f).read()
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True) if _new(p)]
assert src, ('Old code found: upload the newest xfer-code.zip as a new version of Speech emotion VLM, then in the '
             'Input panel open its menu and choose Check for updates (or remove it and add it again)')
CODE = '/kaggle/working/xfer'
os.chdir('/kaggle/working')
if os.path.exists(CODE):
    shutil.rmtree(CODE)            # always start from the newest code, even in a session that ran before
shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt catboost "gradio>=5,<6"', shell=True, check=True)
subprocess.run('pip uninstall -y -q torchao', shell=True)      # the old Kaggle torchao breaks peft (step 4)
import importlib.metadata as md
from packaging.version import Version
if Version(md.version('transformers')) < Version('4.52'):
    subprocess.run('pip install -q -U "transformers>=4.52"', shell=True, check=True)
print('code from', src[0])
""")
md("## 1. Find the trained models and the data list")
code("""
import pandas as pd
OUT = '/kaggle/working/outputs'
os.makedirs(f'{OUT}/models', exist_ok=True)
def one(pattern, what, who):
    hits = glob.glob(pattern, recursive=True)
    assert hits, f'{what} not found: add {who} as an input'
    print(f'{what}: {hits[0]}'); return hits[0]
man = one('/kaggle/input/**/notebooke3145f0d3e/**/outputs/manifest.csv', 'data list', 'notebooke3145f0d3e')
shutil.copy2(man, f'{OUT}/manifest.csv')
svm = one('/kaggle/input/**/outputs/models/svm_C.joblib', 'SVM', 'notebooke3145f0d3e')
CKPT = one('/kaggle/input/**/checkpoints_seed42/best.pth', 'POSTER++', 'notebooke3145f0d3e')
A2 = one('/kaggle/input/**/experts/A2_best.pth', 'fine-tuned CLIP', 'the step 5 notebook notebooka8c6aeaeff')
hits = glob.glob('/kaggle/input/**/slm/lora/adapter_config.json', recursive=True)
if hits:
    LORA = os.path.dirname(hits[0]); print('SLM adapter:', LORA)
    LORA_ARG = '--lora ' + LORA
else:
    LORA = ''; LORA_ARG = ''
    print('NOTE: SLM adapter (slm/lora) not found in any input, so the app will run without the SLM check.')
    print('      The final result does not use the SLM. What the inputs contain under an slm folder:')
    for d in glob.glob('/kaggle/input/**/slm', recursive=True): print('     ', d, os.listdir(d)[:10])
WORK = '/kaggle/working/poster'
for sub in ('POSTER_V2', 'pretrain'):
    s = os.path.join(os.path.dirname(os.path.dirname(CKPT)), sub)
    if os.path.isdir(s) and not os.path.exists(os.path.join(WORK, sub)):
        shutil.copytree(s, os.path.join(WORK, sub))
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
md("## 2. Start the app. Models load in about 10 to 15 minutes; the public link (…gradio.live) appears below and this cell keeps running while the app is open.")
code("""
!python scripts/31_demo_app2.py --out {OUT} --work {WORK} --ckpt {CKPT} --a2 {A2} {LORA_ARG} --svm {svm} --share
""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_demo2.ipynb"), "w"), indent=1)
print("wrote")
