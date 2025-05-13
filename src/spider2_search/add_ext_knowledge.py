import json
from pathlib import Path
import lancedb
from tqdm import tqdm

INPUT_FILE = "./data/spider2-lite.jsonl"
DOCUMENTS_DIR = Path("./data/documents")
OUTPUT_FILE = "./data/spider2-lite-with-ext-knowledge.jsonl"


def replace_external_knowledge(input_file, output_file, documents_dir):
    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            if not line.strip():
                continue
            data = json.loads(line)
            ext_know_path = data.get("external_knowledge")

            if ext_know_path:
                doc_path = documents_dir / ext_know_path.split("/")[-1]
                print(doc_path)
                if doc_path.exists():
                    with open(doc_path, "r", encoding="utf-8") as doc_file:
                        data["external_knowledge"] = doc_file.read()
                else:
                    print(f"Warning: {doc_path} not found. Keeping original path.")

            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")


def evaluate_recall_precision(
    input_file,
    db_path,
    chunk_table="chunks",
    top_k=5,
    embed_model="all-MiniLM-L6-v2",
    embedder=None,
):
    # Load data
    queries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "question" in item and "instance_id" in item:
                queries.append((item["instance_id"], item["question"]))

    db = lancedb.connect(db_path)
    table = db.open_table(chunk_table)

    correct = 0
    retrieved_total = 0
    relevant_total = len(queries)

    for instance_id, question in tqdm(queries, desc="Evaluating"):
        query_emb = embedder.embed([question])[0]

        results = table.search(query_emb).limit(top_k).to_list()
        matched = any(row["instance_id"] == instance_id for row in results)

        if matched:
            correct += 1

        retrieved_total += top_k

    recall = correct / relevant_total
    precision = correct / retrieved_total

    print(f"\n📊 Precision@{top_k}: {precision:.4f}")
    print(f"📊 Recall@{top_k}:    {recall:.4f}")


if __name__ == "__main__":
    replace_external_knowledge(INPUT_FILE, OUTPUT_FILE, DOCUMENTS_DIR)
