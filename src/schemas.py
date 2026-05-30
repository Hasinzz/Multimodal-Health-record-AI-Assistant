# src/schemas.py
from dataclasses import asdict

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# ==========================================
# MODEL 1: IMAGE EXPERT SCHEMA
# ==========================================
class Model1Output(BaseModel):
    case_id: str
    modality: str = Field(..., description="Either 'chest_xray' or 'brain_mri'")
    top_predictions: List[str]
    probabilities: Dict[str, float]
    embedding_path: str = Field(..., description="Local path to the saved .npy feature vector")
    patient_summary_text: str

# ==========================================
# MODEL 2: DOCUMENT EXPERT SCHEMA (OCR + NER)
# ==========================================
class Entity(BaseModel):
    text: str
    label: str  # e.g., "DRUG", "DISEASE", "DOSAGE"
    confidence: float

class Model2Output(BaseModel):
    case_id: str
    source_file: str
    raw_text: str
    raw_text_preview: str
    entities: List[Entity]
    structured_data: Optional[Dict[str, str]] = None # For tabular lab data if available
    patient_summary: str

# ==========================================
# MODEL 3: FUSION & RAG SCHEMA
# ==========================================
class RetrievedEvidence(BaseModel):
    source_document: str
    text_chunk: str
    similarity_score: float

class Model3Output(BaseModel):
    case_id: str
    image_findings: str
    text_findings: str
    retrieved_evidence: List[RetrievedEvidence]
    final_summary: str
    doctor_feedback: str


def to_dict(obj):
    return asdict(obj)