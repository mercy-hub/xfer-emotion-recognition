# Detailed run notes (all stages)

Code for *Explainable Facial Emotion Recognition through Ontology-Grounded Fusion of Model Explanations and a
Post-Hoc Vision-Language Compensator* (University of Lagos). This first sprint covers data preparation and
**experiment E1**: each source on its own. Later sprints add explanations (SHAP, LIME, surrogate tree, maps), source D
(landmark geometry), the ontology and reasoner, and the compensator.

| Source | Model | Training | Script |
|---|---|---|---|
| V | LLaVA-1.5-7B, 4-bit, zero-shot | none | `03_run_zero_shot.py --source V` |
| A | CLIP ViT-L/14, zero-shot, prompt ensemble | none | `03_run_zero_shot.py --source A` |
| B | POSTER++ (authors' model code) | train split; best epoch on val | `04_train_poster.py` |
| C | HOG + uniform LBP on a 5 x 5 grid, RBF SVM with Platt scaling | train split; C chosen on val | `02_run_svm.py` |

## Where to run it: Kaggle

Use a Kaggle notebook with **GPU T4 x2** and **Internet on** (phone-verified account). Free Colab has no guaranteed
GPU time and disconnects idle sessions; Kaggle gives a weekly GPU allowance and runs that continue in the background
(*Save Version → Save & Run All*). Check the current limits on Kaggle's own settings page.

1. Zip this folder and upload it as a private Kaggle dataset named `xfer-code`.
2. Create a notebook from `notebooks/kaggle_E1.ipynb` (File → Import Notebook).
3. Add two inputs: your RAF-DB copy and `xfer-code`. Run the cells in order.

## Run order

```
python scripts/01_prepare_data.py --root <RAF-DB folder> --out <OUT>     # CPU, minutes
python scripts/02_run_svm.py      --out <OUT>                            # CPU
python scripts/03_run_zero_shot.py --out <OUT> --source A --splits test,val,train   # GPU
python scripts/03_run_zero_shot.py --out <OUT> --source V --splits test,val         # GPU, slowest
python scripts/04_train_poster.py --out <OUT> --work <WORK> --epochs 60            # GPU, hours
python scripts/05_evaluate_e1.py  --out <OUT> --split test                          # CPU
```

Every GPU step is resumable: run the same command again after a session ends. Use `--limit 16` for a quick smoke
test before a full run.

## Stage 2: explanations, landmarks, calibration, surrogate tree

Notebook `notebooks/kaggle_stage2.ipynb`, used as two Kaggle notebooks (`RUN = 'A'`, then `RUN = 'B'`).

| Script | What it does | Where |
|---|---|---|
| `06_landmarks.py` | MediaPipe Face Landmarker: 478 landmarks, 52 blendshapes, 8 geometric measures (source D), region masks | CPU |
| `07_neutral_check.py` | lets LLaVA answer freely on 40 validation faces per class, to test whether "never neutral" is the model or the scoring | GPU |
| `08_calibrate.py` | temperature and vector (prior) recalibration fitted on validation; the vector map on V is the E4 baseline | CPU |
| `09_region_explain.py` | Kernel SHAP and LIME over 7 facial regions from one shared set of 32 masked images per image; any source; shards across GPUs; resumable | GPU for V |
| `10_surrogate_tree.py` | shallow tree trained to copy V's answers from geometry (and V's SHAP); fidelity by depth; rules as text | CPU |

Kernel SHAP with 32 samples was checked against exact Shapley values on a 7-region test function (largest error 0.017
on values up to 0.45; the values always add up to f(all) - f(none)).

## Stage 3: the reasoner (E2, E3)

`scripts/11_reasoner.py` (CPU, about a minute) uses `xfer/ontology.py` (action-unit configurations, region links,
confusable pairs, blendshape-to-action-unit reading) and `xfer/reasoner.py` (translation, scoring, decision flow,
consistency, traces). Weights and thresholds are fitted on validation; the test split is used once. Outputs:
`results/e2_test.md`, `results/e3_test.csv`, `results/traces_test.csv`, `results/reasoner_params.json`,
`figures/risk_coverage.png`.

## The ontology (OWL 2)

`ontology/xfer_emonto.owl` is the extended EMONTO ontology. The reasoner reads its knowledge from this file
(`scripts/11_reasoner.py --owl`, on by default). Rebuild and check it with:

    python -m xfer.owl build ontology/xfer_emonto.owl
    python -m xfer.owl check ontology/xfer_emonto.owl     # HermiT (needs Java) + six competency questions

## Stage 4: the compensator (E4)

| Script | Arm | Where |
|---|---|---|
| `12_compensator_tabular.py` | detector (all images) + tabular corrector (V's failures only); ablations pred, pred+expl, pred+face, pred+expl+face | CPU |
| `13_compensator_vision.py` | CNN corrector (ConvNeXt-T); ablations image, image+pred, image+pred+attr (4th channel = V's region SHAP map) | 1 GPU |
| `14_compensator_vlm.py` | Qwen2.5-VL-3B with LoRA; ablations image, image+pred, image+pred+expl (prompt) | 1 GPU |
| `15_evaluate_e4.py` | same gate for every arm, threshold on validation, Table 18 with net-correction intervals; writes the compensated V of each image arm | CPU |
| `16_full_framework.py` | the reasoner with V replaced by its compensated output (Table 17, full framework; E3 repeated) | CPU |

Notebook: `notebooks/kaggle_E4.ipynb`. Tests: `python tests/test_stage4.py` (the Qwen arm's model code runs first on Kaggle).

## Data rules (section 3.3 of the report)

* The RAF-DB layout is detected automatically: official release, folder per class (`train/1..7`, `test/1..7`), or
  train/test CSV files. Labels follow RAF-DB ids 1..7 = surprise, fear, disgust, happiness, sadness, anger, neutral.
  **Check `figures/contact_sheet.png`**: if a Kaggle copy used a different order, every row will look wrong.
* The official test set is never changed. The official training set is split ~80/20 into train and val,
  stratified by class, with near-duplicate images (256-bit difference hash, Hamming distance <= 16) kept together.
  Near-duplicates between the official train and test sets are counted and listed, not removed.
* Hyperparameters (SVM C, POSTER++ epoch) are chosen on val; the test set is used once.

## Outputs (in `<OUT>`)

`manifest.csv`, `data_report.json`, `class_counts.csv/.md`, `figures/`, `predictions/<source>_<split>.csv`
(image_id, label, p_surprise ... p_neutral, pred), `results/e1_test.md` (Table 16, single-source rows), confusion
matrices and McNemar tests.

## Notes and known limits

* POSTER++: the official repository selects its best epoch on the test set and hard-codes Windows paths to its
  pretrained backbones; `xfer/sources/poster.py` patches the paths and our own loop selects on val. The repository's
  `save_checkpoint` also writes the optimiser state, not the model, to its "best" file; this is avoided here. The paper
  trains for 200 epochs; 60 is the default here to fit a Kaggle session, so state the number used in Chapter 4.
* LLaVA-1.5 class probabilities use the first token of each class word (Equation 3.2). The script checks that the
  seven first tokens differ and otherwise scores whole words (slower). The mode used is printed at start-up.
* LLaVA-1.5 and CLIP ViT-L/14 share an encoder family, so the reasoner treats V and A as correlated sources.
* Stage 2 is tested with `python tests/test_stage2.py` (calibration, explanations with the real SVM, sharding and
  resume, merge, geometry, surrogate tree). MediaPipe's model file, LLaVA, CLIP and POSTER++ weights could not be
  downloaded in the build environment, so those paths run for the first time on Kaggle.
* Tested here on CPU with a synthetic dataset (`python tests/test_pipeline.py`): data preparation for all three
  layouts, split rules, the SVM, metrics and the E1 report. The POSTER++ model was built and trained for one epoch on
  CPU with randomly initialised backbones. CLIP and LLaVA could not be downloaded in the build environment, so run
  the `--limit` smoke test on Kaggle first.
* RAF-DB is licensed for non-commercial research. A copy taken from Kaggle should be acknowledged as such in the
  report; requesting official access as well is advisable.

## Demo app (step 17)

`scripts/17_demo_app.py` runs the whole framework on one face (upload or webcam) in a Gradio web app: face crop and
regions, the four models' predictions, region SHAP heatmaps, action units, the compensator decision, and the final
label / blend / abstention with its justification trace. On Kaggle use `notebooks/kaggle_demo.ipynb` (Run All in the
editor; it prints a public gradio.live link). Test without a GPU: `python scripts/17_demo_app.py --out OUT --mock`.

## New design, step 1: the new VLM (InternVL3-8B)

The supervisor's revised architecture replaces LLaVA-1.5 with a stronger frozen VLM and puts an SLM compensator after it.

| Script | What it does | Hardware |
|---|---|---|
| `18_run_vlm.py` | frozen VLM (`xfer/sources/vlm_zs.py`, default `OpenGVLab/InternVL3-8B-hf`): `--task proba` (class probabilities, `predictions/V2_<split>.csv`), `--task describe` (structured FACS-style description, emotion not given) and `--task given` (description with the true emotion in the prompt, train and val only: reference explanations for the SLM; refused on test) | GPU |
| `19_eval_vlm.py` | accuracy, macro-F1, ECE and per-class recall of the new VLM against LLaVA-1.5, McNemar, and parse rates of the descriptions (`results/vlm_compare_<split>.md`) | CPU |

Run it with `notebooks/kaggle_step1_vlm.ipynb` (built by `notebooks/make_notebook_step1.py`).

## New design, step 2: the shared explanation (region occlusion)

| Script | What it does | Hardware |
|---|---|---|
| `20_occlusion.py` | blurs each of 10 fine face regions (`xfer/regions.py` `FINE`: brows, eyes and cheeks split by side) and records the drop in every emotion's probability; then deletion and insertion curves (top-1..k regions) for the predicted class as a faithfulness check. Works for V2, A, B, C (and the old V) through `xfer/predictors.py`. Output `explanations/occ_<source>_<split>.npz/.csv` | GPU for V2, A, B |

Run it with `notebooks/kaggle_step2_occlusion.ipynb` (built by `notebooks/make_notebook_step2.py`).

## New design, steps 3 to 6 (CPU) and the remaining boxes (step 4 notebook)

| Script | What it does | Hardware |
|---|---|---|
| `21_new_framework.py` | Step 3 (the VLM's claims checked against MediaPipe, `xfer/claims.py`), Stage 2 fusion (translate, condition on class, weight by calibrated confidence x per-class reliability x faithfulness, aggregate with agreement score and outlier flags), Stage 3 decision (rules read from the OWL file: AU pattern, confusability tie-break, geometry over VLM claims; label, compound label or abstain), traces with the rules that fired, and the fused explanation per test image (`explanations/fused_test.csv`) | CPU |
| `22_instantiate.py` | every test face's evidence and decision as individuals of the EMONTO extension (`<OUT>/ontology/instances_test.owl`), SPARQL checks over them, and a HermiT consistency check and classification on a subset | CPU |
| `23_native_xai.py` | each expert's own explanation: linear SVM exact weights, CLIP relevance maps and AU-phrase probing, POSTER++ cross-attention relevance and Integrated Gradients on the landmark stream, the VLM's answer-token relevance; each compared with the shared occlusion explanation | GPU (linsvm: CPU) |
| `24_slm.py` | the SLM compensator (Qwen2.5-1.5B + LoRA): keep / change / abstain with an explanation, and exact field-level Shapley values | GPU |

The GPU parts and UniFER-7B (`18_run_vlm.py --model Karl28/UniFER-7B --tag U --forms --answer_prefix ...`) run in
`notebooks/kaggle_step4_complete.ipynb` (built by `notebooks/make_notebook_step4.py`).

## Final system and the Phase 2 demo app
* `scripts/30_final_system.py --out OUT` fits the final combiner (CatBoost averaged with a weighted average, on the
  validation split) and the decision thresholds, and stores them in `xfer/final/` (already included).
* `scripts/31_demo_app2.py` is the Phase 2 demo: four experts with native and occlusion explanations, the VLM's claims
  checked against MediaPipe, the SLM compensator, the fused explanation and the EMONTO + FACS decision with its trace.
  Run it on Kaggle with `notebooks/kaggle_demo2.ipynb`; test it without a GPU with `--mock --selftest face.png`.
