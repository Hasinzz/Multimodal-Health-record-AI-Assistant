from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
XRAY_DIR = IMAGE_DIR / "xray"
BRAIN_MRI_DIR = IMAGE_DIR / "brain_mri"
DOCUMENT_DIR = DATA_DIR / "documents"
STRUCTURED_DIR = DATA_DIR / "structured"
KB_DIR = DATA_DIR / "kb"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL1_CHECKPOINT_DIR = CHECKPOINT_DIR / "model1"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

DEFAULT_XRAY_CHECKPOINT = MODEL1_CHECKPOINT_DIR / "xray_best_model.pt"
DEFAULT_BRAIN_CHECKPOINT = MODEL1_CHECKPOINT_DIR / "brain_best_model.pt"

XRAY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

BRAIN_CLASSES = [
    "glioma_tumor",
    "meningioma_tumor",
    "no_tumor",
    "pituitary_tumor",
]


def create_required_folders():
    folders = [
        DATA_DIR,
        IMAGE_DIR,
        XRAY_DIR,
        BRAIN_MRI_DIR,
        DOCUMENT_DIR,
        STRUCTURED_DIR,
        KB_DIR,
        CHECKPOINT_DIR,
        MODEL1_CHECKPOINT_DIR,
        OUTPUT_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)