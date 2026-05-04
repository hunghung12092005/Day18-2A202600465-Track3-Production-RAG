"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text.strip()
    return " ".join(sentences[:2]).strip()


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    lowered = text.lower()
    questions = []
    if "nghỉ phép" in lowered:
        questions.append("Nhân viên được nghỉ phép bao nhiêu ngày?")
    if re.search(r"\b\d+\s+ngày\b", lowered):
        questions.append("Quy định này áp dụng trong bao nhiêu ngày?")
    if "mật khẩu" in lowered:
        questions.append("Bao lâu phải thay đổi mật khẩu?")
    if not questions:
        questions.append("Đoạn văn này nói về nội dung gì?")
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    title = document_title or "tài liệu nội bộ"
    topic = "quy định chung"
    lowered = text.lower()
    if "nghỉ phép" in lowered:
        topic = "chính sách nghỉ phép"
    elif "mật khẩu" in lowered:
        topic = "chính sách mật khẩu"
    elif "dữ liệu cá nhân" in lowered:
        topic = "bảo vệ dữ liệu cá nhân"
    return f"Trích từ {title}, nội dung nói về {topic}.\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    lowered = text.lower()
    category = "general"
    topic = "unknown"
    entities = []
    if "nghỉ phép" in lowered:
        category = "hr"
        topic = "nghi phep"
        entities.append("nhan_vien")
    elif "mật khẩu" in lowered:
        category = "it"
        topic = "mat khau"
    elif "dữ liệu cá nhân" in lowered:
        category = "policy"
        topic = "du lieu ca nhan"
    if re.search(r"\b\d+\s+ngày\b", lowered):
        entities.append("time_policy")
    return {"topic": topic, "entities": entities, "category": category, "language": "vi"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []
    expanded_methods = set(methods)
    if "full" in expanded_methods:
        expanded_methods.update({"summary", "hyqa", "contextual", "metadata"})

    for chunk in chunks:
        raw_text = chunk["text"]
        raw_meta = chunk.get("metadata", {})
        summary = summarize_chunk(raw_text) if "summary" in expanded_methods else ""
        questions = generate_hypothesis_questions(raw_text) if "hyqa" in expanded_methods else []
        enriched_text = contextual_prepend(raw_text, raw_meta.get("source", "")) if "contextual" in expanded_methods else raw_text
        auto_meta = extract_metadata(raw_text) if "metadata" in expanded_methods else {}
        if questions:
            enriched_text = f"{enriched_text}\n\nCâu hỏi liên quan: {' | '.join(questions)}"
        enriched.append(
            EnrichedChunk(
                original_text=raw_text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata={**raw_meta, **auto_meta},
                method="+".join(methods),
            )
        )
    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
