"""Comprehensive test suite for Information Retrieval (IR) quality evaluation (Recall, MRR, nDCG, Runner, Charts, Telemetry)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from dynavec.dashboard import _make_handler
from dynavec.eval import (
    LabeledQuery,
    RetrievalEvalRunner,
    RetrievalEvalSummary,
    RetrievalMetricResult,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    plot_retrieval_metrics,
)
from dynavec.exceptions import MissingDependencyError
from dynavec.models import SearchResult
from dynavec.telemetry import TelemetryRecorder, aggregate, aggregate_eval

# ===========================================================================
# 1. Mathematical Accuracy Tests for IR Metric Algorithms
# ===========================================================================


class TestIRMetricMath:
    def test_recall_at_k(self) -> None:
        retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]
        relevant = {"doc_a", "doc_c"}

        assert compute_recall_at_k(retrieved, relevant, k=1) == 0.5  # 1/2
        assert compute_recall_at_k(retrieved, relevant, k=2) == 0.5  # 1/2
        assert compute_recall_at_k(retrieved, relevant, k=3) == 1.0  # 2/2
        assert compute_recall_at_k(retrieved, relevant, k=10) == 1.0

    def test_recall_edge_cases(self) -> None:
        assert compute_recall_at_k([], {"doc_a"}, k=5) == 0.0
        assert compute_recall_at_k(["doc_a"], set(), k=5) == 0.0
        assert compute_recall_at_k(["doc_a"], {"doc_a"}, k=0) == 0.0

    def test_mrr_scoring(self) -> None:
        retrieved = ["doc_a", "doc_b", "doc_c", "doc_d"]

        # First relevant at rank 1 -> 1/1 = 1.0
        assert compute_mrr(retrieved, {"doc_a", "doc_c"}) == 1.0
        # First relevant at rank 2 -> 1/2 = 0.5
        assert compute_mrr(retrieved, {"doc_b"}) == 0.5
        # First relevant at rank 4 -> 1/4 = 0.25
        assert compute_mrr(retrieved, {"doc_d"}) == 0.25
        # None found in top-2 cutoff -> 0.0
        assert compute_mrr(retrieved, {"doc_c"}, k=2) == 0.0
        # None found anywhere -> 0.0
        assert compute_mrr(retrieved, {"doc_x"}) == 0.0
        assert compute_mrr([], {"doc_a"}) == 0.0

    def test_ndcg_at_k_binary_relevance(self) -> None:
        # Ideal ranking: relevant docs at positions 0 and 1
        retrieved_ideal = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = {"doc_1", "doc_2"}
        assert compute_ndcg_at_k(retrieved_ideal, relevant, k=2) == 1.0
        assert compute_ndcg_at_k(retrieved_ideal, relevant, k=4) == 1.0

        # Suboptimal ranking: relevant docs at positions 1 and 3 (0-indexed)
        retrieved_suboptimal = ["doc_x", "doc_1", "doc_y", "doc_2"]
        score_k2 = compute_ndcg_at_k(retrieved_suboptimal, relevant, k=2)
        score_k4 = compute_ndcg_at_k(retrieved_suboptimal, relevant, k=4)
        assert 0.0 < score_k2 < 1.0
        assert 0.0 < score_k4 < 1.0
        assert score_k4 > score_k2  # second relevant item found at rank 4 increases score

    def test_ndcg_at_k_graded_relevance(self) -> None:
        grades = {"doc_high": 3.0, "doc_med": 2.0, "doc_low": 1.0}

        # Perfect ranking: high (3) -> med (2) -> low (1)
        ideal = ["doc_high", "doc_med", "doc_low", "doc_none"]
        assert compute_ndcg_at_k(ideal, grades, k=3) == 1.0

        # Inverted ranking: low (1) -> med (2) -> high (3)
        inverted = ["doc_low", "doc_med", "doc_high", "doc_none"]
        score = compute_ndcg_at_k(inverted, grades, k=3)
        assert 0.0 < score < 1.0

    def test_ndcg_edge_cases(self) -> None:
        assert compute_ndcg_at_k([], {"doc_a": 1.0}, k=5) == 0.0
        assert compute_ndcg_at_k(["doc_a"], {}, k=5) == 0.0
        assert compute_ndcg_at_k(["doc_a"], {"doc_a": 0.0}, k=5) == 0.0
        assert compute_ndcg_at_k(["doc_a"], {"doc_a": 1.0}, k=0) == 0.0

    def test_precision_at_k(self) -> None:
        retrieved = ["doc_1", "doc_2", "doc_x", "doc_y"]
        relevant = {"doc_1", "doc_2", "doc_3"}

        assert compute_precision_at_k(retrieved, relevant, k=2) == 1.0  # 2/2
        assert compute_precision_at_k(retrieved, relevant, k=4) == 0.5  # 2/4
        assert compute_precision_at_k(retrieved, set(), k=4) == 0.0


# ===========================================================================
# 2. Data Models & Dataset Loading Tests
# ===========================================================================


class TestRetrievalDataModelsAndLoading:
    def test_labeled_query_creation_and_dict(self) -> None:
        lq = LabeledQuery(
            query="serverless vector database",
            relevant_ids=["doc_1", "doc_2"],
            relevance_grades={"doc_1": 2.0, "doc_2": 1.0},
        )
        assert lq.query_id.startswith("q_")
        assert isinstance(lq.relevant_ids, set)
        d = lq.to_dict()
        assert d["query"] == "serverless vector database"
        assert d["relevant_ids"] == ["doc_1", "doc_2"]

    def test_summary_serialization_and_save_json(self, tmp_path) -> None:
        r1 = RetrievalMetricResult(
            query_id="q1",
            query="test query",
            retrieved_ids=["doc_1"],
            recall_at_k={1: 1.0, 5: 1.0},
            mrr=1.0,
            ndcg_at_k={1: 1.0, 5: 1.0},
            precision_at_k={1: 1.0, 5: 0.2},
            latency_ms=15.2,
        )
        summary = RetrievalEvalSummary(
            total_queries=1,
            k_values=[1, 5],
            mean_recall={1: 1.0, 5: 1.0},
            mean_mrr=1.0,
            mean_ndcg={1: 1.0, 5: 1.0},
            mean_precision={1: 1.0, 5: 0.2},
            p50_latency_ms=15.2,
            p95_latency_ms=15.2,
            results=[r1],
        )

        out_path = tmp_path / "eval_summary.json"
        summary.save_json(out_path)

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["total_queries"] == 1
        assert data["mean_mrr"] == 1.0
        assert len(data["results"]) == 1

    def test_load_dataset_json_array(self, tmp_path) -> None:
        dataset = [
            {"query": "q1", "relevant_ids": ["d1", "d2"]},
            {"query": "q2", "relevant_ids": ["d3"], "relevance_grades": {"d3": 3.0}},
        ]
        json_file = tmp_path / "dataset.json"
        json_file.write_text(json.dumps(dataset), encoding="utf-8")

        loaded = RetrievalEvalRunner.load_dataset(json_file)
        assert len(loaded) == 2
        assert loaded[0].query == "q1"
        assert loaded[1].relevance_grades == {"d3": 3.0}

    def test_load_dataset_jsonl(self, tmp_path) -> None:
        lines = [
            json.dumps({"query": "q1", "relevant_ids": ["d1"]}),
            json.dumps({"query": "q2", "relevant_ids": ["d2", "d3"]}),
        ]
        jsonl_file = tmp_path / "dataset.jsonl"
        jsonl_file.write_text("\n".join(lines), encoding="utf-8")

        loaded = RetrievalEvalRunner.load_dataset(jsonl_file)
        assert len(loaded) == 2
        assert "d1" in loaded[0].relevant_ids

    def test_load_dataset_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            RetrievalEvalRunner.load_dataset("non_existent_file.json")


# ===========================================================================
# 3. RetrievalEvalRunner Execution Tests
# ===========================================================================


class TestRetrievalEvalRunner:
    def test_runner_with_callable_backend(self) -> None:
        # Mock backend returning plain string IDs
        def mock_search(query: str, top_k: int) -> list[str]:
            if "dynavec" in query:
                return ["doc_dynavec", "doc_other", "doc_noise"][:top_k]
            return ["doc_random_1", "doc_random_2"][:top_k]

        runner = RetrievalEvalRunner(search_backend=mock_search, k_values=[1, 3])
        dataset = [
            LabeledQuery(query="what is dynavec?", relevant_ids={"doc_dynavec"}),
            LabeledQuery(query="other topic", relevant_ids={"doc_target"}),
        ]

        summary = runner.run(dataset)
        assert summary.total_queries == 2
        assert summary.k_values == [1, 3]
        # Query 1: doc_dynavec at rank 1 (Recall@1=1.0, MRR=1.0)
        # Query 2: doc_target not returned (Recall=0.0, MRR=0.0)
        assert summary.mean_recall[1] == pytest.approx(0.5)
        assert summary.mean_mrr == pytest.approx(0.5)
        assert summary.p50_latency_ms >= 0.0

    def test_runner_with_client_search_results(self) -> None:
        class MockClient:
            def search(
                self, query: str, top_k: int = 10, namespace: str | None = None
            ) -> list[SearchResult]:
                return [
                    SearchResult(id="doc_1", score=0.95, distance=0.05, text="text 1"),
                    SearchResult(id="doc_2", score=0.85, distance=0.15, text="text 2"),
                ][:top_k]

        client = MockClient()
        runner = RetrievalEvalRunner(search_backend=client, k_values=[1, 2])
        dataset = [{"query": "test query", "relevant_ids": ["doc_2"]}]

        summary = runner.run(dataset)
        assert summary.total_queries == 1
        # doc_2 is at rank 2: Recall@1 = 0.0, Recall@2 = 1.0, MRR = 0.5
        assert summary.mean_recall[1] == 0.0
        assert summary.mean_recall[2] == 1.0
        assert summary.mean_mrr == 0.5

    def test_runner_invalid_backend_raises(self) -> None:
        runner = RetrievalEvalRunner(search_backend="not_a_searcher")
        with pytest.raises(TypeError, match="must have a search"):
            runner.run([{"query": "q", "relevant_ids": ["d"]}])


# ===========================================================================
# 4. Visual Charting Tests
# ===========================================================================


class TestPlotRetrievalMetrics:
    def test_plot_missing_dependency_raises(self) -> None:
        summary = RetrievalEvalSummary(
            total_queries=1,
            k_values=[1, 5],
            mean_recall={1: 1.0, 5: 1.0},
            mean_mrr=1.0,
            mean_ndcg={1: 1.0, 5: 1.0},
            mean_precision={1: 1.0, 5: 0.2},
            p50_latency_ms=10.0,
            p95_latency_ms=10.0,
        )

        with patch.dict("sys.modules", {"matplotlib": None}):
            with pytest.raises(MissingDependencyError) as exc_info:
                plot_retrieval_metrics(summary)
            assert "matplotlib" in str(exc_info.value)
            assert "benchmark" in str(exc_info.value)

    def test_plot_mocked_generation(self) -> None:
        summary = RetrievalEvalSummary(
            total_queries=5,
            k_values=[1, 3, 5, 10],
            mean_recall={1: 0.5, 3: 0.7, 5: 0.85, 10: 0.95},
            mean_mrr=0.72,
            mean_ndcg={1: 0.5, 3: 0.68, 5: 0.81, 10: 0.91},
            mean_precision={1: 0.5, 3: 0.35, 5: 0.25, 10: 0.15},
            p50_latency_ms=12.0,
            p95_latency_ms=25.0,
        )

        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_matplotlib = MagicMock()
        mock_matplotlib.pyplot = mock_plt

        with patch.dict(
            "sys.modules", {"matplotlib": mock_matplotlib, "matplotlib.pyplot": mock_plt}
        ):
            saved_path = plot_retrieval_metrics(summary, output_path="retrieval_chart.png")
            assert saved_path == "retrieval_chart.png"
            mock_plt.savefig.assert_called_once_with("retrieval_chart.png", bbox_inches="tight")
            mock_plt.close.assert_called_once_with(mock_fig)

    def test_plot_empty_summary_raises(self) -> None:
        mock_plt = MagicMock()
        with patch.dict("sys.modules", {"matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt}):
            with pytest.raises(ValueError, match="Cannot plot empty retrieval evaluation summary"):
                plot_retrieval_metrics(RetrievalEvalSummary(0, [], {}, 0.0, {}, {}, 0.0, 0.0))


# ===========================================================================
# 5. Telemetry & Dashboard Integration Tests
# ===========================================================================


class TestRetrievalTelemetryAndDashboard:
    def test_telemetry_event_retrieval_fields(self) -> None:
        rec = TelemetryRecorder()
        ev = rec.new_event(
            op="search",
            namespace="docs",
            latency_ms=30.0,
            eval_recall=0.92,
            eval_mrr=1.0,
            eval_ndcg=0.88,
        )
        rec.record(ev)

        assert ev.eval_recall == 0.92
        assert ev.eval_mrr == 1.0
        assert ev.eval_ndcg == 0.88

        d = ev.to_dict()
        assert d["eval_recall"] == 0.92
        assert d["eval_mrr"] == 1.0
        assert d["eval_ndcg"] == 0.88

    def test_aggregate_with_retrieval_scores(self) -> None:
        rec = TelemetryRecorder()
        rec.record(rec.new_event("search", eval_recall=1.0, eval_mrr=1.0, eval_ndcg=0.9))
        rec.record(rec.new_event("search", eval_recall=0.8, eval_mrr=0.5, eval_ndcg=0.7))
        rec.record(rec.new_event("search"))  # unevaluated

        stats = aggregate(rec.snapshot())
        assert stats["eval_recall_mean"] == pytest.approx(0.9)
        assert stats["eval_mrr_mean"] == pytest.approx(0.75)
        assert stats["eval_ndcg_mean"] == pytest.approx(0.8)

    def test_aggregate_eval_with_retrieval_scores(self) -> None:
        rec = TelemetryRecorder()
        rec.record(
            rec.new_event("search", eval_recall=1.0, eval_mrr=1.0, eval_ndcg=0.9)
        )  # pass (>=0.7)
        rec.record(
            rec.new_event("search", eval_recall=0.4, eval_mrr=0.5, eval_ndcg=0.5)
        )  # fail (<0.7)

        summary = aggregate_eval(rec.snapshot())
        assert summary["total_evals"] == 2
        assert summary["mean_recall"] == pytest.approx(0.7)
        assert summary["mean_mrr"] == pytest.approx(0.75)
        assert summary["mean_ndcg"] == pytest.approx(0.7)
        assert summary["pass_rate"] == pytest.approx(0.5)

    def test_dashboard_api_retrieval_eval_endpoints(self) -> None:
        rec = TelemetryRecorder()
        rec.record(
            rec.new_event(
                "search",
                namespace="kb",
                latency_ms=25.0,
                eval_recall=0.95,
                eval_mrr=1.0,
                eval_ndcg=0.92,
            )
        )

        handler_cls = _make_handler(rec)
        handler = handler_cls.__new__(handler_cls)
        handler.wfile = MagicMock()
        sent_data = {}

        def mock_send(code, body, ctype="application/json"):
            sent_data["code"] = code
            sent_data["body"] = json.loads(body)
            sent_data["ctype"] = ctype

        handler._send = mock_send

        # Test GET /api/eval/summary
        handler.path = "/api/eval/summary"
        handler.do_GET()
        assert sent_data["code"] == 200
        assert sent_data["body"]["total_evals"] == 1
        assert sent_data["body"]["mean_recall"] == 0.95
        assert sent_data["body"]["mean_mrr"] == 1.0

        # Test GET /api/eval/runs
        handler.path = "/api/eval/runs"
        handler.do_GET()
        assert sent_data["code"] == 200
        assert len(sent_data["body"]) == 1
        assert sent_data["body"][0]["eval_recall"] == 0.95
        assert sent_data["body"][0]["eval_mrr"] == 1.0
        assert sent_data["body"][0]["eval_ndcg"] == 0.92
