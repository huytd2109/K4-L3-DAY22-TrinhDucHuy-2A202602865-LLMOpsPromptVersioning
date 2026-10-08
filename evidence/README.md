

## RAGAS — kết quả lần chạy cuối

| Metric | V1 | V2 |
|---|---:|---:|
| Faithfulness | 0.9458 | 0.9118 |
| Answer relevancy | 0.9078 | 0.8853 |
| Context recall | 1.0000 | 1.0000 |
| Context precision | 0.9450 | 0.9417 |

Chạy mới 50 QA cho mỗi phiên bản và chấm đủ **400 ô điểm hữu hạn**, không bỏ mẫu, không cần retry trong lần cuối. Hai system prompt giống hệt checkpoint 2 và đều có `{context}`. JSON trong evidence khớp byte-for-byte với `data/ragas_report.json`. Cả hai phiên bản đạt faithfulness ≥ 0.9.

V1 cao hơn V2 khoảng 0.0339 ở faithfulness và 0.0226 ở answer relevancy. Câu trả lời V1 trung bình 43.1 từ, V2 79.9 từ. Một giải thích phù hợp với dữ liệu là V1 trả lời trực tiếp, còn V2 diễn giải dài hơn nên có thêm khẳng định chưa được context hỗ trợ. Ví dụ câu 5 về cross-validation: V1 đạt faithfulness 1.0; V2 đạt 0.7 và thêm nhận xét về giảm overfitting, đảm bảo độ vững của model, trong khi ba context truy xuất không nêu các nhận xét này.

Cả 50 câu dùng context giống nhau giữa V1/V2. Chênh lệch context precision nhỏ không chứng minh prompt cải thiện truy xuất; nó phản ánh kết quả chấm của evaluator trên cùng dữ liệu. Kết quả dùng cùng model để tạo và chấm, trên bộ QA của lab; chưa phải kết luận thống kê trên dataset độc lập. Điểm có thể thay đổi khi chạy lại do đầu ra LLM và evaluator.

- [Faithfulness](https://docs.ragas.io/en/v0.2.6/concepts/metrics/available_metrics/faithfulness/): tỷ lệ khẳng định trong câu trả lời được context hỗ trợ.
- [Answer relevancy](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/): độ liên quan với câu hỏi, dựa trên embedding của các câu hỏi sinh ngược từ câu trả lời.
- [Context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/): mức thông tin trong đáp án chuẩn được context bao phủ.
- [Context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/): mức các context liên quan được xếp ở vị trí cao.
