# Main selected-cohort implementation

This directory supplies the main-comparison methods and their shared dependencies. See the root README for the design limitations and distinction from the supplementary analysis. HAC in the manuscript is implemented in `models/HierarchicalDynamicClustering/`; `models/DynamicClustering/` supplies dependencies used by that hierarchy.

Install `requirements.txt` in a dedicated environment and set `SMM_DATA_CSV` to the authorized prepared dataset. Run from this directory with an explicit configuration, for example:

```powershell
python train_model.py --config_path settings/Configurations/Traditional_ML/LR-config.json
```

This command fits a model. Other retained JSON files identify the reported neural, conventional, autoencoder and sampling settings. No new random-undersampling configuration has been invented where a run-specific configuration was not retained. The contrastive pathway expects pretrained weights at its configured location; those weights are not included. Evaluation-only mode likewise requires corresponding local checkpoints.

Outputs follow the original configuration's `results/` layout and are ignored by Git. Keep these private and move/retain them outside the upload folder before publishing. The public package must contain source only. Tracking calls are serviced by a local no-op module; no tracking account or credential is required.

The outcome-based exclusion, pre-split transformations and test-as-validation behavior document the reported main design. They have intentionally not been replaced by the supplementary design. Binary-prediction ROC-AUC from conventional models and the autoencoder must be labeled balanced accuracy when reporting it.
