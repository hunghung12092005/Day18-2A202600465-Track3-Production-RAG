"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[\]}])", r"\1", raw)
        return json.loads(cleaned)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    def tokens(text: str) -> set[str]:
        return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 1}

    per_question: list[EvalResult] = []
    for question, answer, context_list, ground_truth in zip(questions, answers, contexts, ground_truths):
        answer_tokens = tokens(answer)
        gt_tokens = tokens(ground_truth)
        question_tokens = tokens(question)
        context_tokens = set().union(*(tokens(ctx) for ctx in context_list)) if context_list else set()

        faithfulness = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
        answer_relevancy = len(answer_tokens & (question_tokens | gt_tokens)) / max(len(answer_tokens), 1)
        context_precision = len(context_tokens & gt_tokens) / max(len(context_tokens), 1)
        context_recall = len(context_tokens & gt_tokens) / max(len(gt_tokens), 1)

        per_question.append(
            EvalResult(
                question=question,
                answer=answer,
                contexts=context_list,
                ground_truth=ground_truth,
                faithfulness=round(faithfulness, 4),
                answer_relevancy=round(answer_relevancy, 4),
                context_precision=round(context_precision, 4),
                context_recall=round(context_recall, 4),
            )
        )

    def avg(metric: str) -> float:
        if not per_question:
            return 0.0
        return round(sum(getattr(item, metric) for item in per_question) / len(per_question), 4)

    return {
        "faithfulness": avg("faithfulness"),
        "answer_relevancy": avg("answer_relevancy"),
        "context_precision": avg("context_precision"),
        "context_recall": avg("context_recall"),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    def diagnosis_for(metric: str) -> tuple[str, str]:
        mapping = {
            "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature, ground answers in retrieved context"),
            "context_recall": ("Missing relevant chunks", "Improve chunking, query expansion, or add stronger retrieval"),
            "context_precision": ("Too many irrelevant chunks", "Add reranking, metadata filters, or lower retrieval fan-out"),
            "answer_relevancy": ("Answer does not match question", "Improve answer prompt template and query understanding"),
        }
        return mapping[metric]

    ranked = sorted(
        eval_results,
        key=lambda item: (item.faithfulness + item.answer_relevancy + item.context_precision + item.context_recall) / 4,
    )[:bottom_n]
    failures = []
    for item in ranked:
        metrics = {
            "faithfulness": item.faithfulness,
            "answer_relevancy": item.answer_relevancy,
            "context_precision": item.context_precision,
            "context_recall": item.context_recall,
        }
        worst_metric, score = min(metrics.items(), key=lambda pair: pair[1])
        diagnosis, fix = diagnosis_for(worst_metric)
        failures.append(
            {
                "question": item.question,
                "worst_metric": worst_metric,
                "score": score,
                "diagnosis": diagnosis,
                "suggested_fix": fix,
            }
        )
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
