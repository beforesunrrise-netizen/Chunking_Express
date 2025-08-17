from datasets import load_dataset
import json, os, random

TRAIN_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"

def iter_squad_examples(split_url, seed=42):
    ds = load_dataset(
        "json",
        data_files=split_url,
        field="data",   # SQuAD 구조의 최상위 키
        streaming=False
    )["train"]

    for article in ds:
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qid = qa.get("id")
                question = qa.get("question")
                answers = qa.get("answers", [])
                answer_text = answers[0]["text"] if answers else ""
                if answer_text:
                    yield {
                        "id": qid,
                        "title": article.get("title"),
                        "question": question,
                        "context": context,
                        "answer": answer_text,
                    }

def sample_and_save(n=None):
    out = f"data/rag_squad_train_{n}_samples.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    random.seed(42)
    buf = []
    for ex in iter_squad_examples(TRAIN_URL):
        buf.append(ex)
        if len(buf) >= n:
            break

    with open(out, "w", encoding="utf-8") as f:
        json.dump(buf, f, ensure_ascii=False, indent=2)

    print(f"✅ 저장: {out}  ({len(buf)} samples)")

if __name__ == "__main__":
    sample_and_save(n=5)
