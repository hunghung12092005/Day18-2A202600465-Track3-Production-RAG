# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Dong Manh Hung - 2A202600465  
**Thành viên:** Dong Manh Hung → M1/M2/M3/M4/M5

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.2680 | 0.2553 | -0.0127 |
| Context Precision | 0.1048 | 0.1187 | +0.0139 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

## Bottom Failures

### #1
- **Question:** Bao lâu phải thay đổi mật khẩu?
- **Expected:** Mật khẩu phải thay đổi mỗi 90 ngày.
- **Got:** Chunk trả lời đúng chủ đề nhưng có thêm context phụ không liên quan.
- **Worst metric:** `context_precision = 0.0870`
- **Error Tree:** Output gần đúng → Context chưa đủ sạch → Query OK.
- **Root cause:** top-k retrieval còn kéo theo chunk không thuộc nhóm bảo mật mật khẩu.
- **Suggested fix:** thêm metadata filter theo `category="it"` và siết top-k sau rerank.

### #2
- **Question:** Nghị định 13/2023 nói về nội dung gì?
- **Expected:** Nghị định 13/2023 quy định về bảo vệ dữ liệu cá nhân.
- **Got:** Context đúng nhưng còn pha thêm đoạn quyền của chủ thể dữ liệu.
- **Worst metric:** `context_precision = 0.1277`
- **Error Tree:** Output đúng ý chính → Context hơi rộng → Query OK.
- **Root cause:** hierarchical chunking hiện vẫn trả chunk lớn hơn cần thiết cho câu hỏi factoid ngắn.
- **Suggested fix:** dùng child-to-parent lookup đúng nghĩa thay vì trả trực tiếp enriched child text.

### #3
- **Question:** Nhân viên được nghỉ phép năm bao nhiêu ngày?
- **Expected:** Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm.
- **Got:** Context trả về đúng nội dung nghỉ phép nhưng kèm thêm câu về thâm niên.
- **Worst metric:** `context_precision = 0.1413`
- **Error Tree:** Output đúng phần chính → Context rộng hơn câu hỏi → Query OK.
- **Root cause:** answer hiện lấy thẳng chunk đầu tiên, chưa có generation để rút gọn về fact cần thiết.
- **Suggested fix:** thêm LLM generation hoặc rule-based answer extraction cho câu hỏi số lượng/ngày tháng.

## Case Study (cho presentation)

**Question chọn phân tích:** Bao lâu phải thay đổi mật khẩu?

**Error Tree walkthrough:**
1. Output đúng? Có, nhưng chưa đủ gọn.
2. Context đúng? Có một phần, nhưng lẫn chunk phụ.
3. Query rewrite OK? Có.
4. Fix ở bước: retrieval/rerank và answer generation.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tích hợp cross-encoder thật cho M3.
- Thêm metadata-aware retrieval.
- Dùng LLM generate answer thay vì trả nguyên chunk.
