# Thesis Project State

## Stable Baseline

- Brain MRI final checkpoint: `checkpoints/model1/brain_best_model_gpu_final_v2.pt`
- Brain MRI final metrics: accuracy `0.9375`, macro F1 `0.9359`
- Chest X-ray final checkpoint: `checkpoints/model1/xray_best_model_gpu_large_v2.pt`
- Chest X-ray final metrics: macro AUROC `0.8133`, micro AUROC `0.8377`, tuned macro F1 `0.2862`, tuned micro F1 `0.3366`
- Chest X-ray tuned thresholds: `outputs/training/xray_gpu_large_v2_threshold_tuning/xray_tuned_thresholds.json`
- Final validation folder: `outputs/final_run_100_tuned_v2`
- Final validation cases: `100/100 completed`, `0 failed`, technical success rate `100%`

## Stable Commands

- Brain MRI training baseline: `python -m src.model1.train_brain_mri --epochs 15 --batch-size 16 --learning-rate 0.0001 --image-size 224`
- Chest X-ray training baseline: `python -m src.model1.train_xray`
- Case runner baseline: `python -m src.run_case --image <path> --image_modality xray --document <path>`

## Upgrade Policy

- New advanced modules are experimental and optional.
- Stable checkpoints, baseline commands, and final validation outputs must not be overwritten.
- If a dependency, model, or dataset is missing, the code must fall back to the current working method instead of crashing.
