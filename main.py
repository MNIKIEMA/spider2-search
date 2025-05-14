import argparse
from typing import Any
import json
import datasets
import lancedb
from lancedb.table import Table
import pandas as pd
from tabulate import tabulate
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
from spider2_search.ingest import get_or_create_lancedb_table, preprocess_dataset
from spider2_search.plotting import plot_metrics
# from spider2_search.ingest import get_embeddings

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
        nargs="+",
        choices=["vector", "fts", "hybrid"],
        default=["vector", "hybrid"],
        help="Search mode: vector, fts, or hybrid",
    )
    parser.add_argument("--max-k", type=int, default=25, help="Number of results to retrieve")
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[2, 5, 10, 15, 25],
        help="Number of results to retrieve",
    )
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


def run_evaluation(args, table: Table, model):
    print(
        f"Running evaluation with mode: {args.search_mode}, top-k: {args.top_k}, reranker: {args.use_reranker}"
    )

    # Load test queries and ground truth
    dataset = datasets.load_dataset("json", data_files=args.data_file, split="train")
    dataset = dataset.filter(lambda x: bool(x["external_knowledge"]) and bool(x["question"]))
    # dataset = dataset.map(
    #     get_embeddings, batched=True, fn_kwargs={"model": model, "column": "question"}
    # )
    # dataset.with_format(type="numpy", columns=["vector"], output_all_columns=True)

    top_k = [k for k in args.top_k if k <= args.max_k]
    all_results: list[dict[str, Any]] = []

    cross_encoder = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L6-v2", column="chunk"
    )
    rerankers = [None, cross_encoder]

    for reranker in rerankers:
        reranker_name = "cross-encoder" if reranker else None
        embedding_model_name = (
            "static-retrieval-mrl-en-v1-" + reranker_name
            if reranker_name
            else "static-retrieval-mrl-en-v1"
        )
        for search_mode in args.search_mode:
            for item in tqdm(dataset, desc="Evaluating"):
                gt_instance_id = item["instance_id"]
                question = item["question"]
                results = retrieve(
                    question=question,
                    table=table,
                    max_k=args.max_k,
                    mode=search_mode,
                    reranker=reranker,
                )

                for k in top_k:
                    prediction_ids = [result["id"] for result in results[:k]]
                    mrr = calculate_mrr(prediction_ids, [gt_instance_id])
                    recall = calculate_recall(prediction_ids, [gt_instance_id])
                    precision = calculate_precision(prediction_ids, [gt_instance_id])
                    res = [
                        ("mrr", mrr),
                        ("recall", recall),
                        ("precision", precision),
                    ]

                    for metric, value in res:
                        all_results.append(
                            {
                                "id": gt_instance_id,
                                "question": question,
                                "prediction_ids": prediction_ids,
                                "metric": metric,
                                "k": k,
                                "reranker": reranker_name,
                                "embedding_model": embedding_model_name,
                                "query_type": search_mode,
                                "score": value,
                            }
                        )
    df = pd.DataFrame(all_results)
    agg_df = (
        df.groupby(["embedding_model", "query_type", "k", "metric"])
        .agg(avg_score=("score", "mean"))
        .reset_index()
    )
    print("\n📊 Aggregated Results:")
    print(tabulate(agg_df, headers="keys", tablefmt="grid", showindex=True))  # type: ignore
    plot_metrics(agg_df)
    return all_results


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
        print(f"Content: {result['chunk'][:200]}...")
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
    # dataset = dataset.map(lambda x: {"length": len(x["chunk"].split())})
    # dataset = dataset.sort("length")
    # dataset = dataset.remove_columns("length")
    # dataset = dataset.map(get_embeddings, load_from_cache_file=False, fn_kwargs={"model": model})
    # dataset.with_format(type="arrow", columns=["vector"], output_all_columns=True)
    # Get or create table
    print(f"Getting or creating table {table_name}...")

    print(
        f"Embedding Model: {args.embedding_model}, Model Name: {args.model_name} Top K: {args.top_k}"
    )
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
