from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional
from averitec import Datapoint
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from utils.chat import SimpleJSONChat

import json
import os
import pickle
import numpy as np


@dataclass
class RetrievalResult:
    """Container for retrieved documents with list-like interface."""

    documents: List[Document] = field(default_factory=list)
    images: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = None

    def __iter__(self):
        return iter(self.documents)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]


class Retriever:
    """Base class for document retrieval strategies."""

    def get_ris_results(
        self, datapoint: Datapoint, max_per_image: int = 5, max_images: int = 10
    ) -> List[Any]:
        ris_path = getattr(self, "ris_path", None)
        if ris_path is None:
            return [[] for _ in datapoint.claim_images]
        # if attribute ris_results, initiate it
        if not hasattr(self, "ris_results"):
            with open(
                f"{ris_path}/{datapoint.split}_clean.json", "rb"
            ) as f:
                self.ris_results = json.load(f)
        ris_results = self.ris_results.get(str(datapoint.claim_id), {})
        result = []
        for image in datapoint.claim_images:
            result.append(ris_results.get(image, [])[:max_per_image])
        return result

    def __call__(self, datapoint: Datapoint, *args, **kwargs) -> RetrievalResult:
        raise NotImplementedError


class CustomVectorStoreRetriever(Retriever):
    """Retrieves documents using cosine similarity against a custom vector store format.

    The store per claim is at {path}/{claim_id}/ and contains:
      - chunks.pkl       : list of dicts with 'page_content' and 'metadata'
      - embeddings.npy   : float32 array of shape (N, D)
      - pos_to_id.pkl    : mapping from position index to chunk id (optional)
    """

    def __init__(
        self,
        path: str,
        embeddings: Embeddings = None,
        k: int = 10,
        ris_path: Optional[str] = None,
    ):
        self.path = path
        if embeddings is None:
            embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
        self.embeddings = embeddings
        self.k = k
        self.ris_path = ris_path

    def _cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return matrix_norms @ query_norm

    def __call__(self, datapoint: Datapoint, *args, **kwargs) -> RetrievalResult:
        store_dir = f"{self.path}/{datapoint.claim_id}"
        chunks_path = os.path.join(store_dir, "chunks.pkl")
        embeddings_path = os.path.join(store_dir, "embeddings.npy")
        pos_to_id_path = os.path.join(store_dir, "pos_to_id.pkl")

        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        emb_matrix = np.load(embeddings_path).astype(np.float32)
        with open(pos_to_id_path, "rb") as f:
            pos_to_id = pickle.load(f)

        query_vec = np.array(self.embeddings.embed_query(datapoint.claim), dtype=np.float32)
        scores = self._cosine_similarity(query_vec, emb_matrix)
        top_k_indices = np.argsort(scores)[::-1][: self.k]

        documents = []
        for pos_idx in top_k_indices:
            chunk_id = pos_to_id[int(pos_idx)]
            if isinstance(chunks, dict):
                # Try chunk_id as-is, then int, then str
                if chunk_id in chunks:
                    chunk = chunks[chunk_id]
                elif int(chunk_id) in chunks:
                    chunk = chunks[int(chunk_id)]
                elif str(chunk_id) in chunks:
                    chunk = chunks[str(chunk_id)]
                else:
                    continue
            else:
                chunk = chunks[int(pos_idx)]
            if isinstance(chunk, dict):
                page_content = chunk.get("page_content", chunk.get("text", str(chunk)))
                metadata = chunk.get("metadata", {})
            else:
                page_content = str(chunk)
                metadata = {}
            metadata.setdefault("url", "")
            metadata.setdefault("context_before", "")
            metadata.setdefault("context_after", "")
            documents.append(Document(page_content=page_content, metadata=metadata))

        result = RetrievalResult(documents=documents)
        result.images = self.get_ris_results(datapoint)
        return result


class SubqueryRetriever(Retriever):
    """Multi-query retrieval using LLM-generated subqueries for comprehensive coverage."""

    def __init__(self, retriever: Retriever, k=10, fetch_k=50, subqueries=5, lambda_mult=0.5, model=None):
        if model is None:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.retriever = retriever
        self.k = k
        self.fetch_k = fetch_k
        self.subqueries = subqueries
        self.lambda_mult = lambda_mult
        self.client = SimpleJSONChat(
            model=model,
            system_prompt=f"""You are a professional researcher who receives a factual claim and its metadata (speaker, date) and your goal is to output a set of pertinent Google/Bing search queries that could be used to find relevant sources for proving or debunking such claim. You may also use the metadata if they can be used to disambiguate claim and facilitate fact-checking. Ideally, each query would focus on one aspect of the claim, independent of others. You may produce up to 5 search queries which should cover all relevant aspects of the claim and lead to the most successful source search, take your time and be thorough.\nPlease, you MUST output only the best search queries in the following JSON format:\n```json\n[\n    "<query 1>",\n    "<query 2>",\n    "<query 3>",\n    "<query 4>",\n    "<query 5>"\n]\n```""",
        )

    def get_subqueries(self, datapoint):
        """Generates targeted search queries for different aspects of the claim."""
        return self.client(f"{datapoint.claim} ({datapoint.speaker}, {datapoint.claim_date})") + [
            datapoint.claim
        ]

    def __call__(self, datapoint):
        original_claim = datapoint.claim
        queries = self.get_subqueries(datapoint)
        seen_contents = set()
        documents = []
        for subquery in queries:
            datapoint.claim = subquery
            for document in self.retriever(datapoint):
                if "queries" not in document.metadata:
                    document.metadata["queries"] = []
                document.metadata["queries"].append(subquery)
                key = document.page_content[:200]
                if key not in seen_contents:
                    seen_contents.add(key)
                    documents.append(document)
        datapoint.claim = original_claim
        # Re-rank all collected documents by similarity to original claim and return top-k
        if documents:
            query_vec = np.array(
                self.retriever.embeddings.embed_query(original_claim), dtype=np.float32
            )
            doc_vecs = np.array(
                self.retriever.embeddings.embed_documents([d.page_content for d in documents]),
                dtype=np.float32,
            )
            norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10
            doc_vecs_norm = doc_vecs / norms
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            scores = doc_vecs_norm @ query_norm
            top_indices = np.argsort(scores)[::-1][: self.k]
            results = [documents[i] for i in top_indices]
        else:
            results = []
        return RetrievalResult(results, metadata={"queries": queries})
