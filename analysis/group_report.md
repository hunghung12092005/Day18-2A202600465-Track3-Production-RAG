# Group Report — Lab 18: Production RAG

**Nhóm:** Dong Manh Hung - 2A202600465  
**Ngày:** 2026-05-04

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| Dong Manh Hung | M1: Chunking | ☑ | 12/12 |
| Dong Manh Hung | M2: Hybrid Search | ☑ | 5/5 |
| Dong Manh Hung | M3: Reranking | ☑ | 5/5 |
| Dong Manh Hung | M4: Evaluation | ☑ | 4/4 |
| Dong Manh Hung | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.2680 | 0.2553 | -0.0127 |
| Context Precision | 0.1048 | 0.1187 | +0.0139 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

## Key Findings

1. **Biggest improvement:** `context_precision` tăng nhẹ nhờ hybrid retrieval, rerank fallback và contextual enrichment.
2. **Biggest challenge:** repo gốc thiếu dữ liệu markdown, `test_set.json` lỗi format, và nhiều module chỉ là scaffold với `TODO`.
3. **Surprise finding:** khi không có LLM generation thật, production pipeline vẫn giữ `faithfulness` cao nhưng `answer_relevancy` gần như không tăng vì câu trả lời đang là retrieved context thay vì synthesized answer.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): production cải thiện nhẹ precision, các metric còn lại gần như giữ nguyên do đang dùng heuristic fallback.
2. Biggest win — module nào, tại sao: M2 hybrid search là phần có ích nhất vì nó tạo khung cho lexical + semantic retrieval và cho phép cải thiện precision.
3. Case study — 1 failure, Error Tree walkthrough: câu hỏi về mật khẩu bị giảm `context_precision` vì retrieve dư chunk ngoài chủ đề, cho thấy cần filter/rerank mạnh hơn.
4. Next optimization nếu có thêm 1 giờ: thay heuristic evaluation bằng RAGAS thật, thêm LLM generation, và dùng cross-encoder thật cho M3.
