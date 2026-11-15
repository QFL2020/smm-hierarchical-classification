# Severe maternal morbidity classification

Code accompanying **Conventional and hierarchical neural models for retrospective severe maternal morbidity classification**, by Qingfeng Li, Grayson Gooden and Minghan Chen.

## What is included

| Folder/file | Purpose |
|---|---|
| `selected_cohort/` | Main selected-cohort comparison, including conventional models, autoencoder, neural comparators and HAC/HDC; model settings are in `settings/Configurations/`. |
| `sensitivity/single_lr.py` | Supplementary patient-grouped, single-split logistic regression without profile exclusion. |
| `sensitivity/run_comparison.py` | Final three-fold LR, RF, NN and modified-HDC comparison. |
| `sensitivity/modified_hdc.py` | Final supported-branch growth, checkpoint selection and subtree acceptance algorithm. |
| `sensitivity/neural_runtime.py` | Minimal neural architecture, sampler and optimizer dependencies for that comparison. |
| `sensitivity/protocol.json` | Frozen scientific specifications and evaluation rules. |
| `sensitivity/precision_recall.py` | Figure S1 from saved continuous scores. |
| `sensitivity/cohort_calibration.py` | Source-cohort summaries and Figure S2/calibration assessment from saved scores. |
| `source_manifest.json` | Source provenance and packaging adaptations. |

Only the implementations needed for the reported main and supplementary analyses are included. Pilot analyses, diagnostic experiments, superseded HDC implementations, patient data, fitted models, scores and logs are excluded. Some shared main-model modules contain additional methods because the reported pathways import them.

## Interpretation

The main comparison removes 17,157 non-SMM records whose predictor profiles also occur among SMM records, before record-level splitting. Its preprocessing and validation design have the limitations described in the manuscript. This code documents that design; it is not a leakage-free prospective prediction template. Conventional-model and autoencoder metrics named `roc_auc` in this implementation use binary predictions and correspond to **balanced accuracy**, as labeled in the supplement. Do not combine them with continuous-score AUROC.

The supplementary comparison retains records without profile exclusion, groups partitions by patient, and fits preprocessing only on fitting records. It uses separate checkpoint-selection and pruning subsets. Its final modified hierarchy retained the matched NN root in each fold. This exploratory evaluation followed diagnostic use of the development data; it is not independent external validation.

Source code alone cannot reproduce unavailable run-specific main-comparison outputs or guarantee the published rounded aggregates. Pretrained contrastive weights and fitted checkpoints are not distributed. Seeds are explicit in code for reproducibility; versions and hardware can also affect results.

## Data and private outputs

Obtain authorized HCUP and AHA data under their respective access conditions. This repository does not create the linked study dataset. It expects the prepared CSV, with `SMM`, `VisitLink`, `DSHOSPID`, and the predictors listed in `sensitivity/features.json`. Source-row order matters for exact partition reconstruction. Do not upload data, identifiers, row-linked scores, small-cell summaries, checkpoints or logs to this repository.

In PowerShell, set paths to private locations outside this repository:

```powershell
$env:SMM_DATA_CSV = 'D:\private-data\prepared.csv'
$env:SMM_OUTPUT_ROOT = 'D:\private-results\smm'
$env:SMM_DEVICE = 'cuda:0'
```

## Environments and execution

Use separate virtual environments for `selected_cohort/requirements.txt`, `sensitivity/requirements-single-lr.txt`, and `sensitivity/requirements.txt`. The single-split LR used scikit-learn 1.3.0; the final comparison used 1.4.2. Install a PyTorch build suitable for your platform. The final neural comparison used float64 CUDA computation. CPU execution is selectable but is not a claim of identical runtime or numerical results.

The following commands **run analyses**; they are documentation, not steps executed while preparing this release. From the repository root, after installing the appropriate requirements:

```powershell
python sensitivity/single_lr.py
python sensitivity/run_comparison.py --prepare
python sensitivity/run_comparison.py --fit-fold 1
python sensitivity/run_comparison.py --fit-fold 2
python sensitivity/run_comparison.py --fit-fold 3
python sensitivity/run_comparison.py --evaluate
python sensitivity/precision_recall.py
python sensitivity/cohort_calibration.py
```

Complete all three fitting stages before evaluation. Fitting saves checkpoints and per-model logs/scores under the private output root. Use a fresh output root for a new run. Figure/calibration scripts read the score layout produced by this comparison; they do not retrain predictive models. Calibration intercept/slope estimation fits diagnostic regressions to the saved scores.

For the selected-cohort implementation, see `selected_cohort/README.md`. Neither analysis supplies a clinically validated screening tool.

## Release scope and license

Packaging changes comprise portable file locations, removal of credentials/external tracking, extraction of the used neural dependencies, removal of standalone demos and annotations. Scientific specifications were retained. Static syntax/configuration checks and a CLI/import check were performed; no predictive model was fitted or evaluated when preparing this package. End-to-end numerical equivalence of the portable package has not been tested.

No license has been assigned pending author confirmation. Public visibility does not itself grant an open-source license. Review institutional rights and third-party dependencies before assigning one.
