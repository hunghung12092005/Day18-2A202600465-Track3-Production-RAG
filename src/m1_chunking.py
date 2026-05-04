"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n+", text) if s.strip()]
    if not sentences:
        return []

    def tokenize(sentence: str) -> set[str]:
        return {t for t in re.findall(r"\w+", sentence.lower()) if len(t) > 1}

    def similarity(a: str, b: str) -> float:
        ta, tb = tokenize(a), tokenize(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta | tb), 1)

    def is_heading(sentence: str) -> bool:
        stripped = sentence.strip()
        return stripped.startswith("#") or (len(stripped.split()) <= 5 and stripped == stripped.title())

    groups: list[list[str]] = [[sentences[0]]]
    for sentence in sentences[1:]:
        sim = similarity(groups[-1][-1], sentence)
        if is_heading(groups[-1][-1]) or is_heading(sentence):
            groups[-1].append(sentence)
        elif sim < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "chunk_index": i, "strategy": "semantic"},
        )
        for i, group in enumerate(groups)
        if " ".join(group).strip()
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [], []

    parents: list[Chunk] = []
    children: list[Chunk] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current and current_len + para_len > parent_size:
            parent_text = "\n\n".join(current).strip()
            pid = f"parent_{len(parents)}"
            parents.append(
                Chunk(
                    text=parent_text,
                    metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
                )
            )
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        parent_text = "\n\n".join(current).strip()
        pid = f"parent_{len(parents)}"
        parents.append(
            Chunk(
                text=parent_text,
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            )
        )

    for parent in parents:
        pid = parent.metadata["parent_id"]
        step = max(child_size - max(child_size // 4, 1), 1)
        for start in range(0, len(parent.text), step):
            child_text = parent.text[start:start + child_size].strip()
            if not child_text:
                continue
            children.append(
                Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_type": "child"},
                    parent_id=pid,
                )
            )
            if start + child_size >= len(parent.text):
                break

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    for part in parts:
        if not part:
            continue
        if re.match(r"^#{1,3}\s+", part.strip()):
            if current_header or current_content.strip():
                combined = f"{current_header}\n{current_content}".strip()
                if combined:
                    chunks.append(
                        Chunk(
                            text=combined,
                            metadata={**metadata, "section": current_header or "preamble", "strategy": "structure"},
                        )
                    )
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    combined = f"{current_header}\n{current_content}".strip()
    if combined:
        chunks.append(
            Chunk(
                text=combined,
                metadata={**metadata, "section": current_header or "preamble", "strategy": "structure"},
            )
        )
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def summarize(chunks: list[Chunk]) -> dict:
        lengths = [len(c.text) for c in chunks]
        if not lengths:
            return {"num_chunks": 0, "avg_length": 0, "min_length": 0, "max_length": 0}
        return {
            "num_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths), 2),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    basic_chunks: list[Chunk] = []
    semantic_chunks: list[Chunk] = []
    hierarchical_children: list[Chunk] = []
    structure_chunks: list[Chunk] = []
    parent_count = 0

    for doc in documents:
        basic_chunks.extend(chunk_basic(doc["text"], metadata=doc.get("metadata")))
        semantic_chunks.extend(chunk_semantic(doc["text"], metadata=doc.get("metadata")))
        parents, children = chunk_hierarchical(doc["text"], metadata=doc.get("metadata"))
        parent_count += len(parents)
        hierarchical_children.extend(children)
        structure_chunks.extend(chunk_structure_aware(doc["text"], metadata=doc.get("metadata")))

    return {
        "basic": summarize(basic_chunks),
        "semantic": summarize(semantic_chunks),
        "hierarchical": {**summarize(hierarchical_children), "num_parents": parent_count},
        "structure": summarize(structure_chunks),
    }


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
