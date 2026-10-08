"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy từ Bước 2) ──────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI giải đáp trực tiếp bằng ngôn ngữ của câu hỏi. "
    "Chỉ dùng thông tin trong context, trả lời ngắn gọn trong 2-4 câu và tập trung vào ý chính. "
    "Nếu context chưa đủ để trả lời, hãy nói rõ phần thông tin còn thiếu, không suy đoán."
    "\n\nContext:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia AI, trình bày câu trả lời bằng ngôn ngữ của câu hỏi với giọng chuyên môn rõ ràng. "
    "Đối chiếu các đoạn context để chọn các dữ kiện liên quan, chỉ sử dụng những dữ kiện được cung cấp. "
    "Tổ chức câu trả lời trong 3-5 câu theo trình tự: định nghĩa hoặc kết luận, cơ chế hoặc thành phần, "
    "rồi ý nghĩa hoặc giới hạn nếu context đề cập. "
    "Khi câu hỏi yêu cầu liệt kê hoặc so sánh, dùng danh sách hoặc các mục phân biệt rõ ràng. "
    "Nếu thiếu dữ kiện, nêu giới hạn của context và không bổ sung kiến thức bên ngoài."
    "\n\nContext:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]

    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        out = run_rag(retriever, llm, prompt, qa["question"])

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str, resume: bool = False) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    resume=True: dùng lại điểm đã lưu nếu khớp dữ liệu, chỉ chấm lại ô điểm lỗi.
    Mọi điểm phải hợp lệ trước khi tính trung bình; không bỏ mẫu bị lỗi.

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    details_dir = Path(__file__).parent.parent / "data" / "ragas_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    samples_path = details_dir / f"03_ragas_samples_{version}.json"
    if resume and samples_path.exists():
        records = json.loads(samples_path.read_text(encoding="utf-8"))
        if len(records) != len(rag_results) or any(
            sample["user_input"] != row["question"]
            or sample["response"] != row["answer"]
            or sample["reference"] != row["reference"]
            or sample["retrieved_contexts"] != row["contexts"]
            for sample, row in zip(records, rag_results)
        ):
            raise ValueError("Điểm đã lưu không khớp với dữ liệu RAG cần đánh giá")
        print(f"↻ Tiếp tục từ điểm từng mẫu đã lưu cho {version}")
    else:
        result = evaluate(
            dataset,
            metrics=metrics,
            llm=llm_eval,
            embeddings=emb_eval,
            run_config=RunConfig(timeout=600, max_workers=8),
        )
        result.to_pandas().to_json(samples_path, orient="records", indent=2, force_ascii=False, double_precision=15)
        records = json.loads(samples_path.read_text(encoding="utf-8"))

    # Chỉ chấm lại ô điểm lỗi; giữ nguyên điểm hợp lệ và lưu bằng chứng từng lần retry.
    for metric in metrics:
        for attempt in range(1, 3):
            missing = [i for i, row in enumerate(records) if row[metric.name] is None or not np.isfinite(row[metric.name])]
            if not missing:
                break
            print(f"↻ {version}/{metric.name}: chấm lại {len(missing)} ô điểm lỗi (lần {attempt})")
            retry = evaluate(
                build_ragas_dataset([rag_results[i] for i in missing]),
                metrics=[metric], llm=llm_eval, embeddings=emb_eval,
                run_config=RunConfig(timeout=600, max_workers=4),
            )
            retry_path = samples_path.with_name(f"03_ragas_retry_{version}_{metric.name}_{attempt}.json")
            retry.to_pandas().to_json(retry_path, orient="records", indent=2, force_ascii=False, double_precision=15)
            for index, value in zip(missing, retry[metric.name]):
                records[index][metric.name] = float(value) if value is not None and np.isfinite(value) else None
            samples_path.write_text(json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw = [row[key] for row in records]
        if len(raw) != len(rag_results) or not all(v is not None and np.isfinite(v) for v in raw):
            raise RuntimeError(f"RAGAS {version}/{key}: thiếu điểm hợp lệ; kiểm tra log và {samples_path}")
        scores[key] = float(np.mean(raw))

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main(resume: bool = False):
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    details_dir = Path(__file__).parent.parent / "data" / "ragas_details"
    details_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        v1_results = json.loads((details_dir / "03_rag_outputs_v1.json").read_text(encoding="utf-8"))
        v2_results = json.loads((details_dir / "03_rag_outputs_v2.json").read_text(encoding="utf-8"))
        for rows in [v1_results, v2_results]:
            if len(rows) != len(QA_PAIRS) or any(
                row["question"] != qa["question"] or row["reference"] != qa["reference"]
                or not isinstance(row["contexts"], list) or not row["contexts"]
                or not all(isinstance(context, str) for context in row["contexts"])
                or not isinstance(row["answer"], str) or not row["answer"].strip()
                for row, qa in zip(rows, QA_PAIRS)
            ):
                raise ValueError("Kết quả RAG đã lưu thiếu mẫu hoặc không khớp QA_PAIRS")
        print("↻ Dùng lại 50 câu trả lời V1 và 50 câu trả lời V2 đã tạo trong lần chạy trước")
    else:
        vectorstore = setup_vectorstore()
        # Thu thập kết quả RAG cho cả V1 và V2
        v1_results = collect_rag_outputs(vectorstore, "v1")
        (details_dir / "03_rag_outputs_v1.json").write_text(json.dumps(v1_results, indent=2, ensure_ascii=False), encoding="utf-8")
        v2_results = collect_rag_outputs(vectorstore, "v2")
        (details_dir / "03_rag_outputs_v2.json").write_text(json.dumps(v2_results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1", resume=resume)
    v2_scores = run_ragas_eval(v2_results, "v2", resume=resume)

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "Hòa" if np.isclose(s1, s2, rtol=0, atol=1e-9) else "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main(resume="--resume" in sys.argv[1:])
