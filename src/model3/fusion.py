from typing import Dict, List, Optional


def build_fused_query(
    model1_output: Optional[Dict],
    model2_output: Optional[Dict],
) -> str:
    query_parts = []

    if model1_output:
        query_parts.append("Image findings:")
        query_parts.append(model1_output.get("patient_summary_text", ""))

        top_predictions = model1_output.get("top_predictions", [])

        for item in top_predictions[:5]:
            query_parts.append(
                f"{item.get('label')}: {item.get('probability')}"
            )

    if model2_output:
        query_parts.append("Document findings:")
        query_parts.append(model2_output.get("patient_summary", ""))
        query_parts.append(model2_output.get("raw_text_preview", ""))

        entities = model2_output.get("entities", [])

        for entity in entities[:20]:
            query_parts.append(str(entity))

    return "\n".join(query_parts)


def summarize_retrieved_evidence(retrieved_evidence: List[Dict]) -> str:
    if not retrieved_evidence:
        return "No external knowledge-base evidence was retrieved."

    lines = []

    for index, evidence in enumerate(retrieved_evidence, start=1):
        source = evidence.get("source", "unknown source")
        score = evidence.get("score", 0.0)
        text = evidence.get("text", "")

        lines.append(
            f"Evidence {index}: source={source}, score={score:.3f}. "
            f"Relevant note: {text[:300]}..."
        )

    return "\n".join(lines)


def generate_doctor_feedback(
    model1_output: Optional[Dict],
    model2_output: Optional[Dict],
    retrieved_evidence: List[Dict],
) -> str:
    feedback = []

    feedback.append("This AI-generated output should be reviewed by a qualified clinician.")

    if model1_output:
        feedback.append(
            "Image-based findings are probabilistic and should be verified with clinical context."
        )

    if model2_output:
        feedback.append(
            "OCR-extracted document findings may contain recognition errors, especially with scanned or handwritten reports."
        )

    if retrieved_evidence:
        feedback.append(
            "Retrieved knowledge-base evidence was used to provide additional context."
        )
    else:
        feedback.append(
            "No knowledge-base evidence was available, so the result is based only on model outputs."
        )

    return " ".join(feedback)


def generate_final_summary(
    case_id: str,
    model1_output: Optional[Dict],
    model2_output: Optional[Dict],
    retrieved_evidence: List[Dict],
) -> Dict:
    image_findings = None
    text_findings = None

    summary_parts = []

    summary_parts.append(f"Case ID: {case_id}")

    if model1_output:
        image_findings = model1_output.get("patient_summary_text")
        summary_parts.append(f"Image findings: {image_findings}")
    else:
        summary_parts.append("Image findings: No image input was provided.")

    if model2_output:
        text_findings = model2_output.get("patient_summary")
        summary_parts.append(f"Document findings: {text_findings}")
    else:
        summary_parts.append("Document findings: No document input was provided.")

    evidence_summary = summarize_retrieved_evidence(retrieved_evidence)
    summary_parts.append(f"Retrieved evidence: {evidence_summary}")

    final_summary = "\n\n".join(summary_parts)

    doctor_feedback = generate_doctor_feedback(
        model1_output=model1_output,
        model2_output=model2_output,
        retrieved_evidence=retrieved_evidence,
    )

    return {
        "case_id": case_id,
        "image_findings": image_findings,
        "text_findings": text_findings,
        "retrieved_evidence": retrieved_evidence,
        "final_summary": final_summary,
        "doctor_feedback": doctor_feedback,
    }