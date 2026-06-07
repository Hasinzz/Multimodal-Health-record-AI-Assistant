# UI Demo Guide

This document summarizes the Streamlit assistant interface test run for the thesis project.

## App Launch

Launch the local interface with:

```powershell
C:\Users\T2520824\Miniconda3\envs\thesis_gpu\python.exe -m streamlit run app.py
```

## Tested Flows

The following UI flows were validated against the stable backend without retraining any model:

- Brain MRI image-only
- X-ray image-only with tuned thresholds
- Document OCR/entities
- Image + document fusion

## Final Checkpoints Used

- Brain MRI: `checkpoints/model1/brain_best_model_gpu_final_v2.pt`
- X-ray: `checkpoints/model1/xray_best_model_gpu_large_v2.pt`
- X-ray thresholds: `outputs/training/xray_gpu_large_v2_threshold_tuning/xray_tuned_thresholds.json`

## What The UI Displays

Each flow is designed to surface the key thesis outputs in a readable assistant-style layout:

- image prediction
- OCR text preview
- extracted entities
- patient summary
- retrieved evidence
- doctor feedback
- JSON output

## Thesis Scope Note

The interface is a local prototype for thesis demonstration. It is not a deployed clinical system and should not be used for medical decision-making.

## Screenshot Checklist

Capture these screens for the thesis report:

- Brain MRI result
- X-ray result
- OCR result
- Fusion result
