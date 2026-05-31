# Multimodal Health Record AI Assistant

Local thesis project for multimodal medical inference across image, document, and fusion pipelines.

## What this repo contains

- `src/model1`: image inference for X-ray and brain MRI classification.
- `src/model2`: document/OCR pipeline for prescriptions and lab reports.
- `src/model3`: fusion layer that combines model outputs with the knowledge base.
- `checkpoints/model1`: pretrained model weights used by the image pipeline.
- `data/`: image, document, structured, and knowledge-base inputs.
- `outputs/`: generated run artifacts and summaries.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run a single case:

```bash
python -m src.run_case --case_id case_001 --image path/to/image.png --image_modality xray --document path/to/document.pdf
```

Run a batch over the available local data:

```bash
python -m src.run_main_batch
```

## Notes

- If you provide `--image`, you must also provide `--image_modality` as `xray` or `brain_mri`.
- Generated outputs are written under `outputs/`.
- Model-1 checkpoints are expected under `checkpoints/model1/` unless you pass explicit paths.

## Keep In Sync

Use the sync script to fetch and rebase the latest changes from GitHub onto your local branch:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/sync_repo.ps1
```

Run it whenever another device has pushed updates, before you start working, or after you finish a session.

## Repository hygiene

Keep the virtual environment, Python cache files, notebook checkpoints, and generated run outputs out of Git. Large model artifacts should be committed only if you intentionally want them tracked, otherwise prefer Git LFS or external storage.

## Using Tuned X-ray Thresholds

- **Purpose:** Apply per-class tuned decision thresholds (found by `src.model1.tune_xray_thresholds`) during inference so binary decisions match the tuned operating point.
- **Threshold file:** The tuning script writes a JSON file, for example [outputs/training/xray_gpu_full_threshold_tuning/xray_tuned_thresholds.json](outputs/training/xray_gpu_full_threshold_tuning/xray_tuned_thresholds.json).
- **Run a single-case inference with thresholds:** pass the `--xray_thresholds` flag to the `src.run_case` entrypoint which forwards it into the inference path in [src/run_case.py](src/run_case.py#L1).

Example (use your GPU Python environment):

```powershell
C:\Users\T2520824\Miniconda3\envs\thesis_gpu\python.exe -m src.run_case \
	--case_id case_001 \
	--image images/xray/example.jpg \
	--image_modality xray \
	--xray_checkpoint checkpoints/model1/xray_best_model_gpu_full.pt \
	--xray_thresholds outputs/training/xray_gpu_full_threshold_tuning/xray_tuned_thresholds.json
```

- **Batch/eval:** To re-run validation evaluation with the tuned thresholds use the provided helper `src/model1/eval_xray_with_thresholds.py` which writes `xray_tuned_eval_metrics.json` and per-class CSVs under the output folder you pass.

