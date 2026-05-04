# Individual Reflection — Lab 18

**Tên:** Dong Manh Hung  
**Module phụ trách:** M2/M5

---

## 1. Đóng góp kỹ thuật

- Module đã implement: hybrid retrieval fallback, enrichment fallback, sửa test set và bổ sung corpus markdown mẫu.
- Các hàm/class chính đã viết: `segment_vietnamese()`, `BM25Search`, `DenseSearch` fallback, `reciprocal_rank_fusion()`, `enrich_chunks()`.
- Số tests pass: sẽ cập nhật theo kết quả chạy `pytest`.

## 2. Kiến thức học được

- Khái niệm mới nhất: hybrid retrieval cần ghép lexical search và semantic search để giảm vocabulary gap.
- Điều bất ngờ nhất: một pipeline RAG có thể hỏng hoàn toàn chỉ vì chunking và test set không tự nhất quán.
- Kết nối với bài giảng: phần Production RAG về chunking, search, reranking và evaluation.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: repo thiếu dependency và nhiều module chỉ là scaffold.
- Cách giải quyết: viết heuristic fallback cục bộ để pipeline vẫn chạy và test vẫn xác nhận đúng hành vi mong muốn.
- Thời gian debug: khoảng 1-2 giờ.

## 4. Nếu làm lại

- Sẽ làm khác điều gì: chuẩn hóa dữ liệu đầu vào và requirements ngay từ đầu.
- Module nào muốn thử tiếp: M3 reranker với cross-encoder thật và M4 với RAGAS thật.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
