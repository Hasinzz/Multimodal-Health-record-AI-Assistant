from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.config import KB_DIR
from src.model3.cross_modal_attention import CrossModalAttentionFusion, fuse_embeddings
from src.model3.fusion import build_fused_query, generate_final_summary
from src.model3.retriever import LocalTfidfRetriever


def _load_json(path: Optional[str]) -> Optional[Dict]:
    if not path:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    return json.loads(path_obj.read_text(encoding="utf-8"))


def _load_embedding_from_output(model_output: Optional[Dict], fallback_dim: int = 512) -> torch.Tensor:
    if not model_output:
        return torch.zeros(fallback_dim, dtype=torch.float32)

    embedding_path = model_output.get("embedding_path")
    if embedding_path and Path(embedding_path).exists():
        embedding = np.load(embedding_path)
        return torch.tensor(embedding, dtype=torch.float32)

    probs = model_output.get("probabilities") or {}
    if probs:
        values = np.array(list(probs.values()), dtype=np.float32)
        if values.size < fallback_dim:
            values = np.pad(values, (0, fallback_dim - values.size))
        return torch.tensor(values[:fallback_dim], dtype=torch.float32)

    return torch.zeros(fallback_dim, dtype=torch.float32)


def _text_embedding_from_document(model2_output: Optional[Dict], fallback_dim: int = 512) -> torch.Tensor:
    if not model2_output:
        return torch.zeros(fallback_dim, dtype=torch.float32)

    text_source = " ".join(
        [
            str(model2_output.get("patient_summary", "")),
            str(model2_output.get("raw_text_preview", "")),
            json.dumps(model2_output.get("entities", []), ensure_ascii=False),
        ]
    )

    vector = np.zeros(fallback_dim, dtype=np.float32)
    for token in text_source.lower().split():
        digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).hexdigest()
        index = int(digest[:8], 16) % fallback_dim
        vector[index] += 1.0

    return torch.tensor(vector, dtype=torch.float32)


def run_advanced_fusion_pipeline(
    case_id: str,
    model1_output: Optional[Dict] = None,
    model2_output: Optional[Dict] = None,
    kb_dir: str = str(KB_DIR),
    top_k: int = 5,
    fusion_mode: str = "advanced",
    fusion_weights: Optional[str] = None,
) -> Dict:
    fused_query = build_fused_query(model1_output=model1_output, model2_output=model2_output)
    retriever = LocalTfidfRetriever(kb_dir=kb_dir)
    retrieved_evidence = retriever.retrieve(query=fused_query, top_k=top_k)

    base_output = generate_final_summary(
        case_id=case_id,
        model1_output=model1_output,
        model2_output=model2_output,
        retrieved_evidence=retrieved_evidence,
    )

    if fusion_mode != "advanced":
        base_output["fusion_mode_used"] = "stable"
        base_output["fusion_status"] = "stable_template_fusion"
        base_output["fused_query"] = fused_query
        return base_output

    image_embedding = _load_embedding_from_output(model1_output)
    text_embedding = _text_embedding_from_document(model2_output)

    if fusion_weights and Path(fusion_weights).exists():
        try:
            embedding_dim = min(image_embedding.numel(), text_embedding.numel())
            fusion_model = CrossModalAttentionFusion(embedding_dim=embedding_dim)
            state_dict = torch.load(fusion_weights, map_location="cpu")
            if isinstance(state_dict, dict):
                fusion_model.load_state_dict(state_dict, strict=False)
            fusion_result = fuse_embeddings(image_embedding, text_embedding, model=fusion_model)
        except Exception:
            fusion_result = fuse_embeddings(image_embedding, text_embedding)
    else:
        fusion_result = fuse_embeddings(image_embedding, text_embedding)

    base_output["fusion_mode_used"] = "advanced"
    base_output["fusion_status"] = fusion_result.status
    base_output["advanced_attention_embedding"] = fusion_result.fused_embedding.detach().cpu().tolist()
    base_output["fused_query"] = fused_query
    return base_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the experimental advanced Model-3 fusion pipeline.")
    parser.add_argument("--case-id", type=str, default="case_001", help="Unique case ID.")
    parser.add_argument("--model1-output", type=str, default=None, help="Optional model1_output JSON path.")
    parser.add_argument("--model2-output", type=str, default=None, help="Optional model2_output JSON path.")
    parser.add_argument("--kb-dir", type=str, default=str(KB_DIR), help="Knowledge base directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks to retrieve.")
    parser.add_argument("--fusion-mode", type=str, choices=["stable", "advanced"], default="advanced")
    parser.add_argument("--fusion-weights", type=str, default=None, help="Optional path to advanced fusion weights.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model1_output = _load_json(args.model1_output)
    model2_output = _load_json(args.model2_output)
    result = run_advanced_fusion_pipeline(
        case_id=args.case_id,
        model1_output=model1_output,
        model2_output=model2_output,
        kb_dir=args.kb_dir,
        top_k=args.top_k,
        fusion_mode=args.fusion_mode,
        fusion_weights=args.fusion_weights,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Saved] {output_path}")

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
