import argparse
import json
import datasets
import lancedb
from sentence_transformers import SentenceTransformer
from lancedb.rerankers import CrossEncoderReranker
from spider2_search.chunking import MarkdownHeaderTextSplitter
from tqdm import tqdm
import torch

from spider2_search.utils import (
    retrieve,
    calculate_mrr,
    calculate_recall,
    calculate_precision,
)
from spider2_search.ingest import get_or_create_lancedb_table, preprocess_dataset, get_embeddings

torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval system with LanceDB")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers",
        help="Name of the embedding model to use",
    )
    parser.add_argument(
        "--search-mode",
        type=str,
        choices=["vector", "fts", "hybrid"],
        default="vector",
        help="Search mode: vector, fts, or hybrid",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Number of results to retrieve")
    parser.add_argument("--use-reranker", action="store_true", help="Whether to use reranker")
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    parser.add_argument("--query", type=str, help="Single query to run (interactive mode)")
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/spider2-lite-with-ext-knowledge.jsonl",
        help="Path to data file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Alibaba-NLP/gte-modernbert-base",
        help="Name of the model to use for embedding",
    )
    return parser.parse_args()


def run_evaluation(args, table, model):
    print(
        f"Running evaluation with mode: {args.search_mode}, top-k: {args.top_k}, reranker: {args.use_reranker}"
    )

    # Load test queries and ground truth
    dataset = datasets.load_dataset("json", data_files=args.data_file, split="train")
    dataset = dataset.filter(lambda x: bool(x["external_knowledge"]) or bool(x["question"]))
    dataset = dataset.map(
        get_embeddings, batched=True, fn_kwargs={"model": model, "column": "question"}
    )
    dataset.with_format(type="numpy", columns=["vector"], output_all_columns=True)

    reranker = None
    if args.use_reranker:
        reranker = CrossEncoderReranker(column="chunk")

    mrr_scores = []
    recall_scores = []
    precision_scores = []

    for item in tqdm(dataset, desc="Evaluating"):
        gt_instance_id = item["id"]
        question = item["question"]

        results = retrieve(
            question=question,
            table=table,
            max_k=args.top_k,
            mode=args.search_mode,
            reranker=reranker,
        )

        prediction_ids = [result["id"] for result in results]

        # Calculate metrics
        mrr = calculate_mrr(prediction_ids, [gt_instance_id])
        recall = calculate_recall(prediction_ids, [gt_instance_id])
        precision = calculate_precision(prediction_ids, [gt_instance_id])

        mrr_scores.append(mrr)
        recall_scores.append(recall)
        precision_scores.append(precision)

    avg_mrr = sum(mrr_scores) / len(mrr_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)

    print("Evaluation Results:")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")

    return {"mrr": avg_mrr, "recall": avg_recall, "precision": avg_precision}


def run_interactive(args, table):
    reranker = None
    if args.use_reranker:
        reranker = CrossEncoderReranker()

    if args.query:
        query = args.query
    else:
        query = input("Enter your query: ")

    print(f"Searching for: {query}")
    results = retrieve(
        question=query,
        table=table,
        max_k=args.top_k,
        mode=args.search_mode,
        reranker=reranker,
    )

    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Instance ID: {result['id']}")
        print(f"Content: {result['chunk'][:200]}...")  # Show first 200 chars

    return results


def main():
    args = parse_args()

    # Initialize chunker
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "header1"),
            ("##", "header2"),
            ("###", "header3"),
            ("####", "header4"),
        ]
    )

    print("Connecting to LanceDB...")
    db = lancedb.connect("./lancedb")

    dataset = datasets.load_dataset("json", data_files=args.data_file, split="train")
    dataset = dataset.filter(lambda x: x["external_knowledge"] is not None)
    dataset = preprocess_dataset(dataset, splitter)  # type: ignore

    # Create table name based on embedding model
    model_name = args.embedding_model.split("/")[-1]
    model = SentenceTransformer(args.model_name)
    table_name = f"chunks_{model_name}"
    dataset = dataset.map(get_embeddings, batched=True, batch_size=5, fn_kwargs={"model": model})

    # Get or create table
    print(f"Getting or creating table {table_name}...")
    dataset.with_format(type="numpy", columns=["vector"], output_all_columns=True)
    table = get_or_create_lancedb_table(
        db=db,
        table_name=table_name,
        all_docs=dataset.to_list(),
        embedding_model=args.embedding_model,
        model_name=args.model_name,
    )

    if args.eval:
        metrics = run_evaluation(args, table, model)

        # Save metrics to file
        with open(
            f"metrics_{args.search_mode}{'_reranked' if args.use_reranker else ''}.json",
            "w",
        ) as f:
            json.dump(metrics, f, indent=2)
    else:
        run_interactive(args, table)


if __name__ == "__main__":
    main()
