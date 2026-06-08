from typing import Dict, Optional

from src.config import KB_DIR
from src.model3.fusion import build_fused_query, generate_final_summary
from src.model3.retriever import LocalTfidfRetriever


def run_fusion_pipeline(
    case_id: str,
    model1_output: Optional[Dict] = None,
    model2_output: Optional[Dict] = None,
    kb_dir: str = str(KB_DIR),
    top_k: int = 5,
) -> Dict:
    fused_query = build_fused_query(
        model1_output=model1_output,
        model2_output=model2_output,
    )

    retriever = LocalTfidfRetriever(kb_dir=kb_dir)

    retrieved_evidence = retriever.retrieve(
        query=fused_query,
        top_k=top_k,
    )

    output = generate_final_summary(
        case_id=case_id,
        model1_output=model1_output,
        model2_output=model2_output,
        retrieved_evidence=retrieved_evidence,
    )

    output["fused_query"] = fused_query
    output["kb_used"] = str(kb_dir)

    return output
