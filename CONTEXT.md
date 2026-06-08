# Thesis Project Context

Project title: Multimodal AI Assistant for Automated Health Record Summarization and Feedback Analysis

Project root: `C:\Users\T2520824\thesis_multimodal_ai`

## V4 Advanced Improvement Plan

V4 is an incremental improvement layer, not a project rebuild. It is not replacing the stable final model yet. V4 is focused on optional advanced Model-2 and Model-3 improvements while keeping the stable selected checkpoints untouched.

The stable `final_v2` / `large_v2` checkpoints remain the final selected models until V4 proves improvement with actual metrics:

- Brain MRI final: `checkpoints/model1/brain_best_model_gpu_final_v2.pt`
- X-ray final: `checkpoints/model1/xray_best_model_gpu_large_v2.pt`
- X-ray final tuned thresholds: `outputs/training/xray_gpu_large_v2_threshold_tuning/xray_tuned_thresholds.json`
- Stable final validation: `outputs/final_run_100_tuned_v2/`

`retrain_v3` was completed successfully, but the old stable checkpoints remained slightly better. V3 outputs also remain untouched:

- Brain MRI v3: `checkpoints/model1/brain_best_model_gpu_retrain_v3.pt`
- X-ray v3: `checkpoints/model1/xray_best_model_gpu_retrain_v3.pt`
- X-ray v3 thresholds: `outputs/training/xray_gpu_retrain_v3_threshold_tuning/xray_tuned_thresholds.json`
- V3 validation: `outputs/final_run_100_retrain_v3/`

V4 focuses on YOLO ROI, BioBERT NER preparation, RAG KB upgrade, and cross-attention readiness. It follows the dataset audit recommendation:

1. Use `data/improvement/model2_lab_reports/lbmaske` for YOLO ROI annotation and OCR improvement.
2. Use `data/improvement/model2_bd_prescriptions` and `data/improvement/model2_ocr_prescriptions` for drug-name support and BioBERT NER weak labeling.
3. Use `data/improvement/model3_medicine_datasets` and `data/improvement/model3_medical_information` to upgrade the RAG knowledge base.
4. Keep cross-modal attention experimental unless a real paired image-text-label dataset is added.

V4 will not fake training. YOLO, BioBERT, and cross-modal attention will only be described as trained after actual training runs and metrics are produced.

Current V4 training boundaries:

- YOLO ROI cannot train until bounding-box labels exist.
- BioBERT can only be weakly supervised until manual labels are corrected.
- Cross-attention cannot train honestly without paired image-text-label data.

V4 outputs are saved under `outputs/v4_advanced_improvement/` and V4 checkpoints use versioned folders under `checkpoints/model2/` or `checkpoints/model3/`. These folders are separate from stable outputs and retrain_v3 outputs.

### V4 Current Status - 2026-06-08

V4 remains an advanced improvement layer for the existing thesis system, not a separate Model-4.

YOLO ROI V4:

- Status: trained as a weakly supervised pseudo-label experiment.
- Data source: 200 pseudo-labeled lab report pages from `data/roi_yolo_v4/`.
- Classes: `patient_info`, `test_table`, `remarks`, `signature_stamp`.
- Full YOLO run folder: `outputs/v4_advanced_improvement/yolo_roi/yolov8n_roi_v4_pseudolabel_full/`.
- YOLO best checkpoint: `outputs/v4_advanced_improvement/yolo_roi/yolov8n_roi_v4_pseudolabel_full/weights/best.pt`.
- Final validation row: precision 0.81817, recall 0.79275, mAP50 0.83567, mAP50-95 0.60411.
- Claim limitation: pseudo-label validation only, not manually annotated ground-truth validation.

BioBERT/BERT NER V4:

- Weak NER dataset exists: 420 train, 90 validation, 90 test records.
- The requested BioBERT model `dmis-lab/biobert-base-cased-v1.1` could not be loaded cleanly in the local Transformers setup, so the completed V4 run uses `bert-base-cased`.
- Final trained checkpoint: `checkpoints/model2/biobert_ner_v4/`.
- Final metrics file: `outputs/v4_advanced_improvement/biobert_ner/ner_metrics_v4.json`.
- Final weak-label validation metrics: precision 1.0, recall 0.99259, Entity-F1 0.99628.
- Claim limitation: weak-label validation only; these are not expert manually corrected clinical NER labels.
- The advanced NER loader now detects the local V4 checkpoint and can run through `ner_engine="biobert"` as the experimental advanced NER path, even though the trained model used is BERT-based.

RAG KB V4:

- V4 KB text files exist under `data/rag_kb_v4/`.
- Medicine CSV sources were processed.
- Excel sources still require `openpyxl` and a rebuild before claiming full Excel inclusion.

Cross-modal attention V4:

- Not honestly trainable yet because no true paired image-text-label dataset exists.

Correct V4 wording for the thesis:

`V4 adds weakly supervised advanced document-processing experiments: YOLOv8n ROI detection trained on automatically generated pseudo-labels and BERT-based NER trained on rule/dictionary-generated weak labels. These modules are experimental improvements and are evaluated against pseudo/weak labels, not expert manual ground truth.`
