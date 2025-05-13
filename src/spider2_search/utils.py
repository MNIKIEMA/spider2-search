import re
from lancedb.table import Table
from lancedb.rerankers import Reranker
from typing import Optional, Literal


def calculate_mrr(predictions: list[str], gt: list[str]):
    mrr = 0
    for label in gt:
        if label in predictions:
            # Find the relevant item that has the smallest index
            mrr = max(mrr, 1 / (predictions.index(label) + 1))
    return mrr


def calculate_recall(predictions: list[str], gt: list[str]):
    # Calculate the proportion of relevant items that were retrieved
    print(f"GT: {gt}")
    print(f"Predictions: {predictions}")
    return len([label for label in gt if label in predictions]) / len(gt)


def calculate_precision(predictions: list[str], gt: list[str]):
    # Calculate the proportion of retrieved items that are relevant
    return len([label for label in gt if label in predictions]) / len(predictions)


def clean_query_for_fts(query: str) -> str:
    """Clean query for FTS search to avoid syntax errors"""
    # Remove special characters and excessive whitespace that might cause FTS syntax errors
    cleaned = re.sub(r"[^\w\s\']", " ", query)  # Keep alphanumeric, spaces, and apostrophes
    cleaned = re.sub(r"\s+", " ", cleaned).strip()  # Normalize whitespace
    return cleaned


def retrieve(
    question: str,
    table: Table,
    max_k=25,
    mode: Literal["vector", "fts", "hybrid"] = "vector",
    reranker: Optional[Reranker] = None,
):
    try:
        if mode == "fts" or mode == "hybrid":
            # For FTS or hybrid search, clean the query to avoid syntax errors
            clean_question = clean_query_for_fts(question)
            results = table.search(
                query=clean_question, vector_column_name=None, query_type=mode
            ).limit(max_k)
        else:
            # For vector search, use the original query
            results = table.search(question, query_type=mode).limit(max_k)

        if reranker:
            results = results.rerank(reranker=reranker)

        return [
            {"instance_id": result["instance_id"], "chunk": result["chunk"]}
            for result in results.to_list()
        ]
    except ValueError as e:
        print(f"Search error with mode {mode}, falling back to vector search: {str(e)}")
        results = table.search(question, query_type="vector").limit(max_k)
        if reranker:
            results = results.rerank(reranker=reranker)

        return [
            {"instance_id": result["instance_id"], "chunk": result["chunk"]}
            for result in results.to_list()
        ]
