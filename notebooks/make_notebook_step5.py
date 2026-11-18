"""Generates notebooks/kaggle_step5_experts.ipynb (run: python notebooks/make_notebook_step5.py)."""
import json
import os

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                               "source": s.strip("\n").splitlines(True)})

md("""
# Step 5: stronger experts other than POSTER++ (POSTER++ is left unchanged as the baseline to beat)

**Inputs** (Add Input): **RAF-DB DATASET**, **Speech emotion VLM** (newest version of the code) and the notebook
**notebooke3145f0d3e**.
**Settings**: Accelerator **GPU T4 x2**, **Internet on**. Then **Save Version → Save & Run All** (background run,
about 2 to 3 hours; you can close the browser).

Runs at the same time:
* GPU 0: **CLIP ViT-L/14 fine-tuned** on the RAF-DB training images, starting from its zero-shot emotion classifier
  (tag **A2**);
* GPU 1: first **HSEmotion** (EfficientNet-B2 trained on AffectNet, a different dataset, used as it is; tags **H7**,
  **H8**; a few minutes), then the **hard-case specialist**, ConvNeXt-Tiny with a class-balanced focal loss and
  lighting, pose and crop augmentation (tag **S**).

Both choose their best epoch on validation and predict the test set once. Download **step5_results.zip** at the end.
""")

code("""
import os, glob, shutil, subprocess, sys, time
src = [p for p in glob.glob('/kaggle/input/**/xfer/__init__.py', recursive=True)
       if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(p)), 'scripts', '29_hsemotion.py'))]
assert src, 'Old code found: upload the newest xfer-code.zip as a new version of your code dataset (Speech emotion VLM)'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
CODE = '/kaggle/working/xfer'
if not os.path.exists(CODE):
    shutil.copytree(os.path.dirname(os.path.dirname(src[0])), CODE)
os.chdir(CODE); sys.path.insert(0, CODE)
subprocess.run('pip install -q -r requirements-kaggle.txt', shell=True, check=True)
import torch; print('GPUs:', torch.cuda.device_count())
""")

md("## Collect the data list")
code("""
import pandas as pd
OUT = '/kaggle/working/outputs'
for man in glob.glob('/kaggle/input/**/outputs/manifest.csv', recursive=True):
    os.makedirs(f'{OUT}/predictions', exist_ok=True)
    shutil.copy2(man, f'{OUT}/manifest.csv'); print('manifest from', man); break
assert os.path.exists(f'{OUT}/manifest.csv'), 'add notebooke3145f0d3e as an input'
m = pd.read_csv(f'{OUT}/manifest.csv')
if not os.path.exists(m.path.iloc[0]):
    idx = {}
    for dp, _, fs in os.walk('/kaggle/input'):
        for f in fs:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                idx.setdefault(f, os.path.join(dp, f))
    m['path'] = m.path.map(lambda p: idx.get(os.path.basename(p), p))
    m.to_csv(f'{OUT}/manifest.csv', index=False)
print('images found:', m.path.map(os.path.exists).mean(), '| per split:', m.split.value_counts().to_dict())
""")

md("## Quick tests (a few images, one epoch each)")
code("""
T = '/kaggle/working/_smoke'
shutil.rmtree(T, ignore_errors=True); os.makedirs(f'{T}/predictions')
shutil.copy2(f'{OUT}/manifest.csv', T)
for name, cmd in [('CLIP', f'CUDA_VISIBLE_DEVICES=0 python scripts/28_finetune_expert.py --out {T} --arch clip --epochs 1 --limit 64 --work {T}/w'),
                  ('HSEmotion', f'CUDA_VISIBLE_DEVICES=1 python scripts/29_hsemotion.py --out {T} --limit 16'),
                  ('specialist', f'CUDA_VISIBLE_DEVICES=1 python scripts/28_finetune_expert.py --out {T} --arch convnext --epochs 1 --limit 64 --work {T}/w')]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f'--- {name}: exit {r.returncode}'); print(r.stdout[-800:])
    if r.returncode: print(r.stderr[-2500:])
    assert r.returncode == 0, name
shutil.rmtree(T)
print('quick tests passed')
""")

md("## Full training (both GPUs at the same time)")
code("""
jobs = {'clip': 'CUDA_VISIBLE_DEVICES=0 python scripts/28_finetune_expert.py --out {o} --arch clip'.format(o=OUT),
        'specialist': ('CUDA_VISIBLE_DEVICES=1 python scripts/29_hsemotion.py --out {o} ; '
                       'CUDA_VISIBLE_DEVICES=1 python scripts/28_finetune_expert.py --out {o} --arch convnext').format(o=OUT)}
procs = {k: subprocess.Popen(v, shell=True, stdout=open(f'/kaggle/working/log_{k}.txt', 'w'), stderr=subprocess.STDOUT)
         for k, v in jobs.items()}
while any(p.poll() is None for p in procs.values()):
    time.sleep(600)
    for k, p in procs.items():
        ep = [l for l in open(f'/kaggle/working/log_{k}.txt').read().splitlines() if 'epoch' in l or ': acc' in l]
        print(f'[{k}]', 'running' if p.poll() is None else f'finished ({p.returncode})', '|', ep[-1] if ep else '', flush=True)
for k, p in procs.items():
    print(f'===== {k} (exit {p.returncode})')
    print('\\n'.join(l for l in open(f'/kaggle/working/log_{k}.txt').read().splitlines() if 'epoch' in l or ': acc' in l or 'Error' in l or 'AffectNet' in l))
""")

md("## Results and download")
code("""
!cd /kaggle/working && zip -qr step5_results.zip outputs/predictions/A2_* outputs/predictions/S_* outputs/predictions/H7_* outputs/predictions/H8_* log_clip.txt log_specialist.txt
print('download /kaggle/working/step5_results.zip from the Output tab and keep it for the analysis')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_step5_experts.ipynb"), "w"), indent=1)
print("wrote")
