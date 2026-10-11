"""从 CRUD-RAG 原始问答构建固定 30 问检索子集。"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .download_crud import DEFAULT_ROOT, REPOSITORY, REVISION

DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "data" / "crud_rag" / "benchmark.json"
TASKS = ("questanswer_1doc", "questanswer_2docs", "questanswer_3docs")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical(text):
    return "".join(text.split())


def build_dataset(raw, *, seed=20261010, corpus_size=1000):
    """保留原始问答；相关性标签来自上游 news1/news2/news3（文档级银标）。"""
    rng = random.Random(seed)
    selected = []
    events = set()
    questions_seen = set()
    for evidence_count, task in enumerate(TASKS, 1):
        candidates = [(index, record) for index, record in enumerate(raw[task])
                      if record["ID"] not in events
                      and isinstance(record.get("questions"), str)
                      and record["questions"].strip()
                      and isinstance(record.get("answers"), str)
                      and record["answers"].strip()
                      and all(isinstance(record.get(f"news{j}"), str)
                              and len(record[f"news{j}"].strip()) > 80
                              for j in range(1, evidence_count + 1))]
        rng.shuffle(candidates)
        count = 0
        for index, record in candidates:
            if record["ID"] in events or record["questions"] in questions_seen:
                continue
            selected.append((task, index, record, evidence_count))
            events.add(record["ID"])
            questions_seen.add(record["questions"])
            count += 1
            if count == 10:
                break
        if count != 10:
            raise ValueError(f"{task} 不足 10 个有效且事件独立的样本")

    documents = {}
    questions = []

    def add_document(text, task, index, record, field):
        doc_id = "news_" + digest(canonical(text))[:24]
        origin = {"task": task, "record_index": index, "event_id": record["ID"], "field": field}
        if doc_id not in documents:
            documents[doc_id] = {"id": doc_id, "text": text, "origins": [origin]}
        elif origin not in documents[doc_id]["origins"]:
            documents[doc_id]["origins"].append(origin)
        return doc_id

    for task, index, record, evidence_count in selected:
        relevant = [add_document(record[f"news{j}"], task, index, record, f"news{j}")
                    for j in range(1, evidence_count + 1)]
        questions.append({"id": f"Q{len(questions) + 1:02d}", "task": task,
                          "upstream_record_index": index, "event_id": record["ID"],
                          "question": record["questions"], "reference_answer": record["answers"],
                          "relevant_document_ids": list(dict.fromkeys(relevant))})
    evidence_ids = set(documents)
    if corpus_size < len(documents):
        raise ValueError("corpus_size 不能小于参考新闻数")
    candidates = []
    positive_texts = [canonical(document["text"]) for document in documents.values()]
    seen = set(positive_texts)
    for task in TASKS:
        for index, record in enumerate(raw[task]):
            if record["ID"] in events:
                continue
            for field in ("news1", "news2", "news3"):
                text = record.get(field)
                if not isinstance(text, str) or len(text.strip()) <= 80:
                    continue
                normalized = canonical(text)
                if normalized in seen:
                    continue
                # 不把相同/包含关系的参考新闻放进未标注的干扰集合。
                if any(normalized in positive or positive in normalized for positive in positive_texts):
                    continue
                seen.add(normalized)
                candidates.append((text, task, index, record, field))
    rng.shuffle(candidates)
    for text, task, index, record, field in candidates:
        if len(documents) == corpus_size:
            break
        add_document(text, task, index, record, field)
    if len(documents) != corpus_size:
        raise ValueError(f"去重后只有 {len(documents)} 篇新闻，无法构建 {corpus_size} 篇语料")
    shuffled_documents = list(documents.values())
    rng.shuffle(shuffled_documents)
    result = {
        "schema_version": 1,
        "provenance": {"repository": REPOSITORY, "revision": REVISION,
                       "dataset_path": "data/crud_split/split_merged.json",
                       "seed": seed, "question_count": 30, "corpus_size": corpus_size,
                       "evidence_document_count": len(evidence_ids),
                       "distractor_document_count": corpus_size - len(evidence_ids),
                       "relevance": "upstream news fields; document-level silver labels",
                       "license_note": "No LICENSE file found in the pinned upstream tree. "
                                       "Data retained locally for this evaluation; verify upstream terms before redistribution."},
        "questions": questions,
        "documents": shuffled_documents,
    }
    result["dataset_sha256"] = digest(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


def validate_dataset(dataset):
    if len(dataset["questions"]) != 30:
        raise ValueError("评测集必须包含 30 个问题")
    docs = {doc["id"]: doc for doc in dataset["documents"]}
    if len(docs) != len(dataset["documents"]):
        raise ValueError("文档 ID 重复")
    seen_questions = set()
    seen_ids = set()
    for question in dataset["questions"]:
        if question["id"] in seen_ids or question["question"] in seen_questions:
            raise ValueError("问题 ID 或问题文本重复")
        seen_ids.add(question["id"])
        seen_questions.add(question["question"])
        gold = set(question["relevant_document_ids"])
        if not gold or not gold <= docs.keys():
            raise ValueError("参考文档缺失")
    expected = dataset["dataset_sha256"]
    actual = digest(json.dumps({key: value for key, value in dataset.items()
                               if key != "dataset_sha256"}, ensure_ascii=False, sort_keys=True))
    if actual != expected:
        raise ValueError("评测集摘要不匹配；请重新 prepare 或同步更新数据版本")
    return dataset


def load_dataset(path=DEFAULT_DATASET):
    return validate_dataset(json.loads(Path(path).read_text(encoding="utf-8")))


def render_questions(dataset):
    lines = ["# CRUD-RAG 固定 30 问", "",
             f"抽样种子：{dataset['provenance']['seed']}。官方单篇、两篇、三篇新闻任务各 10 问；问题保持原文。", ""]
    for question in dataset["questions"]:
        lines.extend([f"## {question['id']} · {question['task']} · "
                      f"{len(question['relevant_document_ids'])} 篇去重参考新闻", "",
                      question["question"], "", f"参考答案：{question['reference_answer']}", "",
                      f"来源记录：{question['event_id']}；原始索引：{question['upstream_record_index']}", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_ROOT / "data/crud_split/split_merged.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--corpus-size", type=int, default=1000)
    args = parser.parse_args()
    raw_bytes = args.raw.read_bytes()
    dataset = build_dataset(json.loads(raw_bytes), seed=args.seed, corpus_size=args.corpus_size)
    dataset.pop("dataset_sha256")
    dataset["provenance"]["upstream_file_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    dataset["dataset_sha256"] = digest(json.dumps(dataset, ensure_ascii=False, sort_keys=True))
    validate_dataset(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_name("questions.md").write_text(render_questions(dataset), encoding="utf-8")
    print(json.dumps(dataset["provenance"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
