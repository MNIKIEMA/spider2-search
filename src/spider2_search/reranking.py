from typing import Optional
import pyarrow as pa
from lancedb.rerankers import Reranker
from functools import cached_property
from sentence_transformers import CrossEncoder
import torch


class ONNXCrossEncoderReranker(Reranker):
    """
    A custom reranker for LanceDB that uses an ONNX backend for cross-encoder models.

    This reranker provides:
    1. Increased performance through ONNX runtime
    2. Flexibility to filter results based on criteria
    3. Support for various cross-encoder models
    4. Batch processing for efficiency
    """

    def __init__(
        self,
        model_name: str,
        max_length: int = 256,
        batch_size: int = 32,
        device: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        trust_remote_code: bool = True,
        column: str = "text",
        **kwargs,
    ):
        """
        Initialize the ONNX Cross-Encoder Reranker.

        Args:
            model_name_or_path: Original model name or path for tokenization
            onnx_model_path: Path to the ONNX model file
            max_length: Maximum sequence length for tokenization
            filters: String or list of strings to filter out from results
            batch_size: Number of examples to process at once
            score_threshold: Minimum score threshold for results
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        super().__init__(**kwargs)

        self.model_name = model_name
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.max_length = max_length
        self.column = column
        self.device = device
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @cached_property
    def model(self):
        # Allows overriding the automatically selected device
        cross_encoder = CrossEncoder(
            model_name_or_path=self.model_name,
            backend="onnx",
            device=self.device,
            model_kwargs=self.model_kwargs,
        )

        return cross_encoder

    def _rerank(self, result_set: pa.Table, query: str):
        result_set = self._handle_empty_results(result_set)
        if len(result_set) == 0:
            return result_set
        passages = result_set[self.column].to_pylist()
        cross_inp = [[query, passage] for passage in passages]
        cross_scores = self.model.predict(cross_inp)
        result_set = result_set.append_column(
            "_relevance_score", pa.array(cross_scores, type=pa.float32())
        )

        return result_set

    def rerank_hybrid(
        self,
        query: str,
        vector_results: pa.Table,
        fts_results: pa.Table,
    ):
        combined_results = self.merge_results(vector_results, fts_results)
        combined_results = self._rerank(combined_results, query)
        # sort the results by _score
        if self.score == "relevance":
            combined_results = self._keep_relevance_score(combined_results)
        elif self.score == "all":
            raise NotImplementedError("return_score='all' not implemented for CrossEncoderReranker")
        combined_results = combined_results.sort_by([("_relevance_score", "descending")])

        return combined_results

    def rerank_vector(self, query: str, vector_results: pa.Table):
        vector_results = self._rerank(vector_results, query)
        if self.score == "relevance":
            vector_results = vector_results.drop_columns(["_distance"])

        vector_results = vector_results.sort_by([("_relevance_score", "descending")])
        return vector_results

    def rerank_fts(self, query: str, fts_results: pa.Table):
        fts_results = self._rerank(fts_results, query)
        if self.score == "relevance":
            fts_results = fts_results.drop_columns(["_score"])

        fts_results = fts_results.sort_by([("_relevance_score", "descending")])
        return fts_results
