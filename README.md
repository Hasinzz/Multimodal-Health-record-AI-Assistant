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

## Repository hygiene

Keep the virtual environment, Python cache files, notebook checkpoints, and generated run outputs out of Git. Large model artifacts should be committed only if you intentionally want them tracked, otherwise prefer Git LFS or external storage.
