"""Information Retrieval (IR) quality evaluation runner (Recall@k, MRR, nDCG@k, Precision@k)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LabeledQuery:
    """A ground-truth labeled evaluation query."""

    query: str
    relevant_ids: set[str] | list[str]
    query_id: str = ""
    relevance_grades: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_id:
            self.query_id = f"q_{abs(hash(self.query)) % 1_000_000:06d}"
        if isinstance(self.relevant_ids, list):
            self.relevant_ids = set(self.relevant_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "relevant_ids": sorted(self.relevant_ids),
            "relevance_grades": self.relevance_grades,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalMetricResult:
    """Per-query retrieval metric results across multiple k cutoffs."""

    query_id: str
    query: str
    retrieved_ids: list[str]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "retrieved_ids": self.retrieved_ids,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "precision_at_k": self.precision_at_k,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalEvalSummary:
    """Dataset-level aggregated summary of retrieval quality metrics."""

    total_queries: int
    k_values: list[int]
    mean_recall: dict[int, float]
    mean_mrr: float
    mean_ndcg: dict[int, float]
    mean_precision: dict[int, float]
    p50_latency_ms: float
    p95_latency_ms: float
    results: list[RetrievalMetricResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "k_values": self.k_values,
            "mean_recall": self.mean_recall,
            "mean_mrr": self.mean_mrr,
            "mean_ndcg": self.mean_ndcg,
            "mean_precision": self.mean_precision,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, file_path: str | Path) -> None:
        """Save the summary results as formatted JSON."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ===========================================================================
# Mathematical IR Metric Algorithms (Pure Math, Zero Dependencies)
# ===========================================================================


def compute_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Recall@k: fraction of all relevant documents retrieved in top-k.

    Recall@k = |Retrieved_k ∩ Relevant| / |Relevant|
    """
    rel_set = set(relevant_ids)
    if not rel_set or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in rel_set)
    return round(hits / len(rel_set), 4)


def compute_mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int | None = None,
) -> float:
    """Compute Mean Reciprocal Rank (MRR): 1 / rank of the first relevant document.

    If no relevant document is retrieved within top-k, returns 0.0.
    """
    rel_set = set(relevant_ids)
    if not rel_set:
        return 0.0

    candidates = retrieved_ids[:k] if k is not None and k > 0 else retrieved_ids
    for rank, doc_id in enumerate(candidates, start=1):
        if doc_id in rel_set:
            return round(1.0 / rank, 4)
    return 0.0


def compute_ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_grades_or_ids: Mapping[str, float] | Set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@k).

    Supports binary relevance (sets/lists) or graded relevance (dict of doc_id -> grade).
    nDCG@k = DCG@k / IDCG@k
    where DCG@k = sum_{i=1}^k (2^rel_i - 1) / log2(i + 1)
    """
    if k <= 0:
        return 0.0

    if isinstance(relevant_grades_or_ids, Mapping):
        grades = {str(doc_id): float(grade) for doc_id, grade in relevant_grades_or_ids.items()}
    else:
        grades = {str(doc_id): 1.0 for doc_id in relevant_grades_or_ids}

    if not grades or all(g <= 0.0 for g in grades.values()):
        return 0.0

    # 1. Compute DCG@k
    dcg = 0.0
    cutoff = min(k, len(retrieved_ids))
    for i in range(cutoff):
        doc_id = str(retrieved_ids[i])
        rel = grades.get(doc_id, 0.0)
        if rel > 0.0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 2.0)

    # 2. Compute Ideal DCG (IDCG@k)
    sorted_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(sorted_grades):
        if rel > 0.0:
            idcg += (math.pow(2.0, rel) - 1.0) / math.log2(i + 2.0)

    if idcg <= 0.0:
        return 0.0

    return round(min(1.0, dcg / idcg), 4)


def compute_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Set[str] | Sequence[str],
    k: int,
) -> float:
    """Compute Precision@k: fraction of top-k retrieved documents that are relevant.

    Precision@k = |Retrieved_k ∩ Relevant| / k
    """
    rel_set = set(relevant_ids)
    if not rel_set or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in rel_set)
    return round(hits / float(k), 4)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac, 2)


# ===========================================================================
# Batch Retrieval Evaluation Runner
# ===========================================================================


class RetrievalEvalRunner:
    """Executes Information Retrieval (IR) evaluation over benchmark datasets.

    Parameters
    ----------
    search_backend:
        Either a Dynavec client instance with ``.search(query, top_k=...)``
        or any custom callable taking ``(query, top_k)`` and returning a list of
        document IDs or SearchResult objects.
    k_values:
        Sequence of rank cutoffs to evaluate (default: [1, 3, 5, 10]).
    """

    def __init__(
        self,
        search_backend: Any,
        k_values: Sequence[int] = (1, 3, 5, 10),
    ) -> None:
        self.search_backend = search_backend
        self.k_values = sorted(set(k_values))
        self.max_k = max(self.k_values) if self.k_values else 10

    def _execute_search(self, query: str, top_k: int, namespace: str | None = None) -> list[str]:
        """Execute search and extract list of document ID strings."""
        if hasattr(self.search_backend, "search"):
            kwargs: dict[str, Any] = {"top_k": top_k}
            if namespace:
                kwargs["namespace"] = namespace
            raw_hits = self.search_backend.search(query, **kwargs)
        elif callable(self.search_backend):
            raw_hits = self.search_backend(query, top_k)
        else:
            raise TypeError("search_backend must have a search() method or be callable.")

        doc_ids: list[str] = []
        for hit in raw_hits:
            if hasattr(hit, "id"):
                doc_ids.append(str(hit.id))
            elif isinstance(hit, dict) and "id" in hit:
                doc_ids.append(str(hit["id"]))
            elif isinstance(hit, str):
                doc_ids.append(hit)
            else:
                doc_ids.append(str(hit))
        return doc_ids

    def evaluate_query(
        self,
        labeled_query: LabeledQuery | dict[str, Any],
        namespace: str | None = None,
    ) -> RetrievalMetricResult:
        """Evaluate a single labeled query against the search backend."""
        if isinstance(labeled_query, dict):
            lq = LabeledQuery(
                query=labeled_query["query"],
                relevant_ids=labeled_query["relevant_ids"],
                query_id=labeled_query.get("query_id", ""),
                relevance_grades=labeled_query.get("relevance_grades"),
                metadata=labeled_query.get("metadata", {}),
            )
        else:
            lq = labeled_query

        t0 = time.perf_counter()
        retrieved_ids = self._execute_search(lq.query, top_k=self.max_k, namespace=namespace)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        rel_target: Any = lq.relevance_grades if lq.relevance_grades else lq.relevant_ids

        recall_map: dict[int, float] = {}
        ndcg_map: dict[int, float] = {}
        precision_map: dict[int, float] = {}

        for k in self.k_values:
            recall_map[k] = compute_recall_at_k(retrieved_ids, lq.relevant_ids, k)
            ndcg_map[k] = compute_ndcg_at_k(retrieved_ids, rel_target, k)
            precision_map[k] = compute_precision_at_k(retrieved_ids, lq.relevant_ids, k)

        mrr_score = compute_mrr(retrieved_ids, lq.relevant_ids, k=self.max_k)

        return RetrievalMetricResult(
            query_id=lq.query_id,
            query=lq.query,
            retrieved_ids=retrieved_ids,
            recall_at_k=recall_map,
            mrr=mrr_score,
            ndcg_at_k=ndcg_map,
            precision_at_k=precision_map,
            latency_ms=round(elapsed_ms, 2),
            metadata=lq.metadata,
        )

    def run(
        self,
        dataset: Sequence[LabeledQuery | dict[str, Any]] | str | Path,
        namespace: str | None = None,
    ) -> RetrievalEvalSummary:
        """Run retrieval evaluation over an entire dataset."""
        loaded_queries = self.load_dataset(dataset) if isinstance(dataset, (str, Path)) else dataset

        results: list[RetrievalMetricResult] = []
        latencies: list[float] = []

        recall_sums = {k: 0.0 for k in self.k_values}
        ndcg_sums = {k: 0.0 for k in self.k_values}
        precision_sums = {k: 0.0 for k in self.k_values}
        mrr_sum = 0.0

        for item in loaded_queries:
            metric_res = self.evaluate_query(item, namespace=namespace)
            results.append(metric_res)
            latencies.append(metric_res.latency_ms)

            mrr_sum += metric_res.mrr
            for k in self.k_values:
                recall_sums[k] += metric_res.recall_at_k.get(k, 0.0)
                ndcg_sums[k] += metric_res.ndcg_at_k.get(k, 0.0)
                precision_sums[k] += metric_res.precision_at_k.get(k, 0.0)

        n = len(results)
        mean_recall = {k: round(recall_sums[k] / n, 4) if n > 0 else 0.0 for k in self.k_values}
        mean_ndcg = {k: round(ndcg_sums[k] / n, 4) if n > 0 else 0.0 for k in self.k_values}
        mean_precision = {
            k: round(precision_sums[k] / n, 4) if n > 0 else 0.0 for k in self.k_values
        }
        mean_mrr = round(mrr_sum / n, 4) if n > 0 else 0.0

        sorted_lat = sorted(latencies)
        p50_lat = _percentile(sorted_lat, 50)
        p95_lat = _percentile(sorted_lat, 95)

        return RetrievalEvalSummary(
            total_queries=n,
            k_values=self.k_values,
            mean_recall=mean_recall,
            mean_mrr=mean_mrr,
            mean_ndcg=mean_ndcg,
            mean_precision=mean_precision,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            results=results,
        )

    @staticmethod
    def load_dataset(
        source: str | Path | Sequence[dict[str, Any] | LabeledQuery],
    ) -> list[LabeledQuery]:
        """Load benchmark queries from JSON, JSONL, or a Python sequence."""
        if isinstance(source, (str, Path)):
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"Dataset file not found: {p}")

            text = p.read_text(encoding="utf-8").strip()
            queries: list[LabeledQuery] = []

            # 1. Try standard JSON array
            if text.startswith("["):
                raw_data = json.loads(text)
                for item in raw_data:
                    queries.append(
                        LabeledQuery(
                            query=item["query"],
                            relevant_ids=item["relevant_ids"],
                            query_id=item.get("query_id", ""),
                            relevance_grades=item.get("relevance_grades"),
                            metadata=item.get("metadata", {}),
                        )
                    )
                return queries

            # 2. Try JSONL (line-delimited JSON)
            for line in text.splitlines():
                clean = line.strip()
                if clean:
                    item = json.loads(clean)
                    queries.append(
                        LabeledQuery(
                            query=item["query"],
                            relevant_ids=item["relevant_ids"],
                            query_id=item.get("query_id", ""),
                            relevance_grades=item.get("relevance_grades"),
                            metadata=item.get("metadata", {}),
                        )
                    )
            return queries

        out: list[LabeledQuery] = []
        for item in source:
            if isinstance(item, LabeledQuery):
                out.append(item)
            elif isinstance(item, dict):
                out.append(
                    LabeledQuery(
                        query=item["query"],
                        relevant_ids=item["relevant_ids"],
                        query_id=item.get("query_id", ""),
                        relevance_grades=item.get("relevance_grades"),
                        metadata=item.get("metadata", {}),
                    )
                )
        return out
