import typing as t
import datasets
from lancedb.embeddings import get_registry
from lancedb.db import DBConnection
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from spider2_search.chunking import MarkdownHeaderTextSplitter, chunk_with_md_header


def get_func(name: str, model: str) -> EmbeddingFunction:
    func: EmbeddingFunction = get_registry().get(name)
    if func is None:
        raise ValueError(f"Function {name} not found in registry")
    return func.create(name=model)


def preprocess_dataset(
    data: datasets.Dataset, splitter: MarkdownHeaderTextSplitter
) -> datasets.Dataset:
    """
    Load and format the external knowledge dataset.

    Returns:
        A list of dictionaries containing instance_id and external_knowledge.
    """

    # Filter and format the dataset in one step
    formatted_dataset = [
        {
            "instance_id": item["instance_id"],
            "external_knowledge": item["external_knowledge"],
        }
        for item in data.to_list()
        if item.get("external_knowledge")
    ]

    all_docs: t.List[t.Dict[str, str]] = []
    for item in formatted_dataset:
        kg = item["external_knowledge"]
        instance_id = item["instance_id"]
        splited_kg = chunk_with_md_header(kg, splitter)
        for chunk in splited_kg:
            all_docs.append({"instance_id": instance_id, "chunk": chunk.page_content})
    return datasets.Dataset.from_list(all_docs)


def get_or_create_lancedb_table(
    db: DBConnection,
    table_name: str,
    all_docs,
    embedding_model: str = "sentence-transformers",
    model_name: str = "all-MiniLM-L6-v2",
):
    func = get_func(name=embedding_model, model=model_name)

    class Chunk(LanceModel):
        instance_id: str
        chunk: str = func.SourceField()
        vector: Vector = func.VectorField()  # type: ignore

    if table_name in db.table_names():
        print(f"Table {table_name} already exists")
        table = db.open_table(table_name)
        table.create_fts_index("chunk", replace=True)
        return table

    table = db.create_table(table_name, mode="overwrite", data=all_docs)
    print(f"Table {table_name} created with {len(all_docs)} chunks")
    table.create_fts_index("chunk", replace=True)
    print(f"{table.count_rows()} chunks ingested into the database")
    return table


def get_embeddings(batch, model: SentenceTransformer, column: str = "chunk"):
    """Get embeddings for a batch of text using the specified model
    Usage:
        >>> dataset = datasets.load_dataset("json", data_files=["./dataset.json"], split="train")
        >>> dataset = dataset.map(get_embeddings, batched=True, fn_kwargs={"model": model})
        >>> dataset.save_to_disk("dataset_with_embeddings")
        >>> dataset = dataset.with_format(type="numpy", columns=["embeddings"], output_all_columns=True)
        >>> table = lance.create_table("dataset_with_embeddings", schema=Chunk)
        >>> table.add(dataset)

    """
    embeddings = model.encode(batch[column])
    return {"vector": embeddings}
