# Explainable Facial Emotion Recognition (XFER)

Source code for the MSc project **"Explainable Facial Emotion Recognition through Ontology-Grounded Fusion of Model Explanations and a Post-Hoc Vision-Language Compensator"**.

- **Author:** Oluwasola Tolase Mercy (249074221)
- **Supervisor:** Dr. E. P. Fasina, Machine Intelligence Research Group
- **Institution:** Department of Computer Science, University of Lagos

## What the system does

The system recognises one of seven emotions (surprise, fear, disgust, happiness, sadness, anger, neutral) from a still photo of a face, and explains how it reached its answer. It works in three stages.

1. **Stage 1, the experts.** Several independent models each give an answer and their own explanation:
   - POSTER++;
   - CLIP ViT-L/14, fine-tuned and zero-shot;
   - InternVL3-8B, a vision-language model that is prompted, never trained;
   - an SVM on HOG and LBP features.

   A shared region-occlusion test checks every explanation in the same way. MediaPipe measures facial action units, and these are used to check the vision-language model's claims about the face. A small language model (Qwen2.5-1.5B with LoRA) acts as a compensator that checks and explains the vision-language model.
2. **Stage 2, XAI fusion.** The evidence is fused. A combiner fitted only on the validation split (CatBoost averaged with a log-linear pool) gives the class probabilities.
3. **Stage 3, symbolic reasoning.** The evidence is instantiated in an OWL 2 extension of the EMONTO ontology with action units and confusable emotion pairs. The decision is one of three outputs, each with a justification trace:
   - a single label;
   - a compound label, such as "fear or surprise";
   - a referral to a person.

## Main results (RAF-DB test set, 3,068 images)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Final system (experts combined) | **94.07%** | **0.904** |
| CLIP fine-tuned (best single expert) | 92.54% | 0.877 |
| POSTER++ | 91.10% | 0.861 |
| InternVL3-8B | 79.69% | 0.685 |
| HOG/LBP + SVM | 72.59% | 0.574 |
| CLIP zero-shot | 55.28% | 0.495 |

- **Accepted images:** the decision stage accepts 96.0% of test images, with an error of 3.8% on them.
- **Compound labels:** 97.8% of them contain the true emotion.
- **Compensators:**
  - With a weak vision-language model (LLaVA-1.5), the compensators raised its accuracy by up to 21.7 points.
  - With a strong one (InternVL3-8B), they raised it by up to 4.2 points.

Every setting was chosen on a validation split taken from the training set. The official test set was used once and was never cleaned or relabelled, and no model was told the true emotion of a test image.

## Repository layout

| Folder | Contents |
|---|---|
| `xfer/` | The core package: data and splits, the experts (`sources/`), occlusion and native explanations, VLM claim checking, the ontology reader, the reasoner, compensators, metrics, and the demo app back end (`demo2.py`). `xfer/final/` holds the fitted combiner (five CatBoost seeds and `final.json`). |
| `scripts/` | Numbered scripts for each step, run in order (`01_prepare_data.py` to `32_svlm_v2.py`). |
| `notebooks/` | Kaggle notebooks for the GPU steps, and the scripts that generate them. |
| `ontology/` | `xfer_emonto.owl`, the EMONTO ontology extended with action units, facial regions and confusable pairs. |
| `bundle/` | Saved vision-language model outputs used to train the compensators (predictions and text only, no images). |
| `tests/` | Small tests that run the pipeline on a synthetic copy of the dataset (`make_fake_raf.py`). |
| `docs/RUNNING.md` | Detailed run notes for every stage and experiment. |

## How to run

The GPU steps were run on Kaggle (GPU T4 x2, internet on). Before running anything, install the extra packages:

```
pip install -r requirements-kaggle.txt
```

**Dataset.** RAF-DB (basic expression subset) is not included. Request it from its authors and pass its folder with `--root`.

```
python scripts/01_prepare_data.py --root <RAF-DB folder> --out <OUT>
```

**Step order.** The steps follow the numbering in `scripts/`. `docs/RUNNING.md` explains each step, its inputs and its outputs, and which notebook runs it.

**Final system.**

```
python scripts/30_final_system.py --out <OUT>
```

This fits the combiner and the decision thresholds on the validation split.

**Demo web app.** The app analyses an uploaded photo and shows every stage of the decision. It takes about 15 seconds per photo on a T4 GPU. To run it on Kaggle, use `notebooks/kaggle_demo2.ipynb`. To test it without a GPU:

```
python scripts/31_demo_app2.py --out <OUT> --mock --selftest face.png
```

**Tests on synthetic data.** The test builds its own small fake copy of the dataset:

```
python tests/test_pipeline.py
```

## Third-party models and data

- POSTER++ model code and pretrained backbones are downloaded from the authors' release at run time.
- CLIP, InternVL3-8B, LLaVA-1.5, Qwen2.5 and MediaPipe are loaded from their public releases.
- RAF-DB is used under its own licence, and its images are not redistributed here.
