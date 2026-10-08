# Bằng chứng bài lab Day 22 — Trịnh Đức Huy

Lần kiểm tra cuối ngày 08/10/2026 chạy mới cả bốn bước bằng `src/run_all.py`: **4/4 PASS, exit code 0**. Model tạo câu trả lời và evaluator: `gpt-4o-mini`; embeddings: `text-embedding-3-small`; RAGAS `0.4.3`; Guardrails `0.11.0`. Chunk size 500, overlap 50, truy xuất k=3.

## RAG và Prompt Hub

LangSmith API xác nhận riêng lần chạy cuối có 50 `rag-query` và 50 `ab-rag-query`, đều hoàn tất không lỗi. Trace mẫu chứa retriever, prompt, LLM, parser và 3 đoạn context.

Hai prompt là `trinh-duc-huy-rag-prompt-v1` và `trinh-duc-huy-rag-prompt-v2`. Cả hai được pull từ Hub, không dùng fallback local. Routing MD5 cho `req-0000`–`req-0049` vẫn là **V1=19, V2=31**. Khi push lại prompt không đổi, Hub trả `409 Nothing to commit`; đây không phải lỗi pull. `02_ab_routing_log.txt` giữ log lần push đầu thành công và đủ 50 nhãn routing.

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

## Guardrails

Hai log demo được lưu từ cùng một lần chạy thực tế: **PII 6/6, JSON 5/5**. PII gồm email, phone, SSN, credit card, nhiều PII và case sạch. JSON gồm hợp lệ, markdown fences, nháy đơn, dấu phẩy thừa và JSON không sửa được.

Validator trả `FailResult(fix_value=...)` khi cần sửa; `OnFailAction.FIX` thay output bằng giá trị đó. `PassResult()` giữ nguyên input sạch. `on_fail` được truyền vào constructor validator để cấu hình cách xử lý lỗi; `Guard.use()` gắn instance đã cấu hình. JSON không sửa được vẫn trả JSON dự phòng có `error` và tối đa 200 ký tự `raw`. Demo kiểm tra output thật trước khi báo thành công. Cảnh báo exporter telemetry của Guardrails không ảnh hưởng các kết quả này.

## Bộ nộp và dữ liệu local

Thư mục này chỉ giữ 7 file bắt buộc theo `SUBMISSION.md` và README phân tích. Log/debug bổ sung được chuyển vào `.local_artifacts/`; câu trả lời và điểm từng mẫu ở `data/ragas_details/`, đều được Git ignore. `.env` và virtualenv cũng không được đưa vào bộ nộp; tài liệu đề bài gốc được giữ nguyên.

`03_ragas_scores.png` chụp trực tiếp log console gốc mở bằng Playwright; văn bản hiển thị được đối chiếu khớp file log. Đây là ảnh log trong trình duyệt, không phải cửa sổ terminal Windows vì công cụ chụp native lỗi.

Chạy toàn bộ từ thư mục gốc:

```powershell
$env:PYTHONUTF8 = "1"
$env:RAGAS_DO_NOT_TRACK = "true"
.\venv\Scripts\python.exe -X utf8 -u src/run_all.py
Copy-Item data/ragas_report.json evidence/03_ragas_report.json
```

Để tiếp tục phần chấm sau lỗi kết nối: `src/03_ragas_evaluation.py --resume`, chỉ dùng với cùng QA, knowledge base, prompt và dữ liệu câu trả lời đã lưu. Chế độ này kiểm tra các trường dữ liệu khớp cache, chấm lại ô lỗi và yêu cầu đủ điểm hợp lệ trước khi tính trung bình.
