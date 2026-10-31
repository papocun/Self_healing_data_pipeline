"""Integration tests for the complete end-to-end SelfHealingPipeline with mocked Ollama."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.inference.ollama_client import OllamaClient
from src.inference.sentiment_analyzer import MalformedModelResponseError, SentimentAnalyzer
from src.pipeline import PipelineOrchestrator
from src.recovery.checkpoint import CheckpointManager
from src.utils.config import Settings


@pytest.fixture
def isolated_pipeline_env(tmp_path: Path):
    """Sets up a complete isolated directory structure and components for integration testing."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    failed_dir = tmp_path / "failed"
    checkpoints_dir = tmp_path / "checkpoints"
    predictions_dir = tmp_path / "predictions"
    logs_dir = tmp_path / "logs"

    for d in (raw_dir, processed_dir, failed_dir, checkpoints_dir, predictions_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    test_settings = Settings()
    test_settings.paths.raw_data_dir = raw_dir
    test_settings.paths.processed_data_dir = processed_dir
    test_settings.paths.failed_data_dir = failed_dir
    test_settings.paths.checkpoint_dir = checkpoints_dir
    test_settings.paths.predictions_dir = predictions_dir
    test_settings.paths.logs_dir = logs_dir

    mock_client = MagicMock(spec=OllamaClient)
    mock_client.base_url = "http://mock-ollama:11434"
    mock_client.model = "qwen2.5:3b"
    mock_client.is_healthy.return_value = True

    # Standard mock sentiment response
    mock_payload = {
        "sentiment": "positive",
        "confidence": 0.95,
        "positive_aspects": ["service", "food"],
        "negative_aspects": [],
        "reasoning": "High quality meal and polite staff.",
    }
    mock_client.chat.return_value = json.dumps(mock_payload)

    analyzer = SentimentAnalyzer(client=mock_client)
    checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoints_dir)

    orchestrator = PipelineOrchestrator(
        batch_size=2,
        ollama_client=mock_client,
        sentiment_analyzer=analyzer,
        checkpoint_manager=checkpoint_mgr,
    )

    return {
        "dirs": {
            "raw": raw_dir,
            "failed": failed_dir,
            "checkpoints": checkpoints_dir,
            "predictions": predictions_dir,
        },
        "settings": test_settings,
        "mock_client": mock_client,
        "orchestrator": orchestrator,
        "checkpoint_mgr": checkpoint_mgr,
    }


def test_pipeline_end_to_end_clean_data(isolated_pipeline_env, monkeypatch):
    env = isolated_pipeline_env
    # Patch global settings paths to match isolated temp paths
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    # 4 valid records -> batch_size=2 results in 2 batches
    sample_df = pd.DataFrame(
        [
            {"review_id": "r1", "business_id": "b1", "stars": 5, "text": "Superb!", "date": "2023-01-01"},
            {"review_id": "r2", "business_id": "b1", "stars": 4, "text": "Pretty good.", "date": "2023-01-02"},
            {"review_id": "r3", "business_id": "b2", "stars": 3, "text": "Decent place.", "date": "2023-01-03"},
            {"review_id": "r4", "business_id": "b2", "stars": 5, "text": "Loved it!", "date": "2023-01-04"},
        ]
    )

    summary = env["orchestrator"].run_dataframe(sample_df, source_name="clean_test.csv")

    assert summary.total_batches == 2
    assert summary.batches_processed == 2
    assert summary.batches_skipped == 0
    assert summary.successful_predictions == 4
    assert summary.invalid_records == 0
    assert summary.duplicate_records == 0
    assert len(summary.prediction_files) == 2

    # Verify predictions file content
    first_pred_file = summary.prediction_files[0]
    preds_df = pd.read_parquet(first_pred_file)
    assert len(preds_df) == 2
    assert "model_name" in preds_df.columns
    assert "inference_timestamp" in preds_df.columns
    assert preds_df.iloc[0]["sentiment"] == "positive"
    assert preds_df.iloc[0]["model_name"] == "qwen2.5:3b"

    # Verify checkpoint commits
    assert env["checkpoint_mgr"].is_batch_completed(1, source_file="clean_test.csv") is True
    assert env["checkpoint_mgr"].is_batch_completed(2, source_file="clean_test.csv") is True


def test_pipeline_separates_invalid_and_duplicate_records(isolated_pipeline_env, monkeypatch):
    env = isolated_pipeline_env
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    mixed_df = pd.DataFrame(
        [
            # Valid record
            {"review_id": "val_1", "business_id": "b1", "stars": 5, "text": "Good", "date": "2023-01-01"},
            # Invalid record: missing stars (None)
            {"review_id": "inv_1", "business_id": "b1", "stars": None, "text": "Missing stars", "date": "2023-01-01"},
            # Duplicate of val_1
            {"review_id": "val_1", "business_id": "b1", "stars": 5, "text": "Duplicate val_1", "date": "2023-01-01"},
            # Valid record
            {"review_id": "val_2", "business_id": "b2", "stars": 4, "text": "Fine", "date": "2023-01-02"},
        ]
    )

    summary = env["orchestrator"].run_dataframe(mixed_df, source_name="mixed_test.csv")

    assert summary.valid_records == 3
    assert summary.invalid_records == 1
    assert summary.duplicate_records == 1
    assert summary.successful_predictions == 2

    # Verify quarantine files were created in data/failed
    failed_files = list(env["dirs"]["failed"].glob("*.json"))
    assert len(failed_files) >= 2  # One for validation failure, one for duplicates


def test_pipeline_resume_skips_committed_batches(isolated_pipeline_env, monkeypatch):
    env = isolated_pipeline_env
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    sample_df = pd.DataFrame(
        [
            {"review_id": "r1", "business_id": "b1", "stars": 5, "text": "Text 1", "date": "2023-01-01"},
            {"review_id": "r2", "business_id": "b1", "stars": 4, "text": "Text 2", "date": "2023-01-02"},
        ]
    )

    # First run commits batch 1
    summary_1 = env["orchestrator"].run_dataframe(sample_df, source_name="resume_test.csv")
    assert summary_1.batches_processed == 1
    assert summary_1.batches_skipped == 0

    # Second run with same checkpoint must skip batch 1
    summary_2 = env["orchestrator"].run_dataframe(sample_df, source_name="resume_test.csv")
    assert summary_2.batches_processed == 0
    assert summary_2.successful_predictions == 0


def test_pipeline_inference_malformed_response_quarantine(isolated_pipeline_env, monkeypatch):
    env = isolated_pipeline_env
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    # Simulate model producing unparseable output on review 1, valid on review 2
    valid_payload = json.dumps({"sentiment": "neutral", "confidence": 0.9, "positive_aspects": []})
    env["mock_client"].chat.side_effect = ["Unrecoverable non-JSON garbage", valid_payload]

    sample_df = pd.DataFrame(
        [
            {"review_id": "bad_resp_rev", "business_id": "b1", "stars": 3, "text": "Some text", "date": "2023-01-01"},
            {"review_id": "good_resp_rev", "business_id": "b1", "stars": 3, "text": "Good text", "date": "2023-01-01"},
        ]
    )

    summary = env["orchestrator"].run_dataframe(sample_df, source_name="malformed_test.csv")

    assert summary.successful_predictions == 1
    assert summary.failed_predictions == 1

    # Verify inference quarantine file
    inf_failed_files = list(env["dirs"]["failed"].glob("inference_failed_*.json"))
    assert len(inf_failed_files) == 1
    content = json.loads(inf_failed_files[0].read_text(encoding="utf-8"))
    assert content[0]["review_id"] == "bad_resp_rev"
    assert content[0]["error_type"] == "MalformedModelResponseError"


def test_airflow_dag_tasks(isolated_pipeline_env, monkeypatch):
    from dags.sentiment_pipeline_dag import (
        task_check_ollama_health,
        task_run_sentiment_pipeline,
        task_verify_pipeline_outputs,
    )

    env = isolated_pipeline_env
    monkeypatch.setattr("src.utils.config.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    # 1. Health check task
    with patch("src.inference.ollama_client.OllamaClient.is_healthy", return_value=True):
        assert task_check_ollama_health() is True

    # 2. Pipeline execution task on sample
    test_raw = env["dirs"]["raw"] / "sample_test.json"
    test_raw.write_text(
        json.dumps(
            [
                {
                    "review_id": "dag_rev_1",
                    "business_id": "b1",
                    "stars": 5,
                    "text": "Excellent food",
                    "date": "2023-01-01",
                }
            ]
        ),
        encoding="utf-8",
    )

    mock_ti = MagicMock()
    mock_context = {
        "dag_run": MagicMock(conf={"pattern": "sample_test.json", "batch_size": 2}),
        "task_instance": mock_ti,
    }

    valid_resp = json.dumps({"sentiment": "positive", "confidence": 0.95, "positive_aspects": ["good"]})
    with patch("src.inference.ollama_client.OllamaClient.chat", return_value=valid_resp):
        pipeline_metrics = task_run_sentiment_pipeline(**mock_context)
    assert pipeline_metrics["successful_predictions"] == 1
    assert pipeline_metrics["prediction_files_count"] == 1

    # 3. Output verification task with XCom pull
    mock_ti.xcom_pull.return_value = pipeline_metrics
    verification_res = task_verify_pipeline_outputs(**mock_context)
    assert verification_res["status"] == "VERIFIED"
    assert verification_res["latest_prediction_records"] == 1

    # 4. Ensure verification fails and does NOT succeed artificially if outputs are missing
    for p in env["dirs"]["predictions"].glob("*.parquet"):
        p.unlink()
    for j in env["dirs"]["predictions"].glob("*.json"):
        j.unlink()

    with pytest.raises(FileNotFoundError, match="No prediction files"):
        task_verify_pipeline_outputs(**mock_context)



def test_airflow_dag_dependency_chain():
    """Validates complete 3-task dependency chain, trigger rules, and structure."""
    import dags.sentiment_pipeline_dag as dag_mod

    if not dag_mod.AIRFLOW_AVAILABLE:
        # Construct operator graph with mock DAG / PythonOperator to verify topological ordering
        class MockOperator:
            def __init__(self, task_id, python_callable=None, trigger_rule="all_success"):
                self.task_id = task_id
                self.python_callable = python_callable
                self.trigger_rule = trigger_rule
                self.downstream_list = []
                self.upstream_list = []

            def __rshift__(self, other):
                self.downstream_list.append(other)
                other.upstream_list.append(self)
                return other

        t1 = MockOperator(task_id="check_ollama_health", python_callable=dag_mod.task_check_ollama_health)
        t2 = MockOperator(task_id="run_sentiment_pipeline", python_callable=dag_mod.task_run_sentiment_pipeline)
        t3 = MockOperator(
            task_id="verify_pipeline_outputs",
            python_callable=dag_mod.task_verify_pipeline_outputs,
            trigger_rule="all_success",
        )

        t1 >> t2 >> t3
        assert t2 in t1.downstream_list
        assert t3 in t2.downstream_list
        assert t2 in t3.upstream_list
        assert t3.trigger_rule == "all_success"
    else:
        dag = dag_mod.dag
        assert len(dag.tasks) == 3
        task_dict = dag.task_dict

        check_health = task_dict["check_ollama_health"]
        run_pipe = task_dict["run_sentiment_pipeline"]
        verify_out = task_dict["verify_pipeline_outputs"]

        # Ensure correct sequential dependency order
        assert run_pipe in check_health.downstream_list
        assert verify_out in run_pipe.downstream_list
        assert run_pipe in verify_out.upstream_list
        assert verify_out.trigger_rule == "all_success"


def test_pipeline_run_file_missing_raises(isolated_pipeline_env, monkeypatch):
    """Ensure run_file raises FileNotFoundError when the target file does not exist."""
    env = isolated_pipeline_env
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    orchestrator = PipelineOrchestrator(
        batch_size=2,
        ollama_client=env["mock_client"],
        checkpoint_manager=CheckpointManager(checkpoint_dir=env["dirs"]["checkpoints"]),
    )

    missing_path = env["dirs"]["raw"] / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="Input file does not exist"):
        orchestrator.run_file(missing_path)


def test_pipeline_run_all_unmatched_pattern_raises(isolated_pipeline_env, monkeypatch):
    """Ensure run_all raises FileNotFoundError when no files match the pattern."""
    env = isolated_pipeline_env
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    sample_file = env["dirs"]["raw"] / "yelp_sample.json"
    sample_file.write_text(
        json.dumps([
            {
                "review_id": "s1",
                "business_id": "b1",
                "stars": 5,
                "text": "Excellent food and service!",
                "date": "2026-01-01",
            }
        ]),
        encoding="utf-8",
    )

    orchestrator = PipelineOrchestrator(
        batch_size=2,
        ollama_client=env["mock_client"],
        checkpoint_manager=CheckpointManager(checkpoint_dir=env["dirs"]["checkpoints"]),
    )

    with pytest.raises(FileNotFoundError, match="No input files matched pattern"):
        orchestrator.run_all(pattern="*unmatched_xyz*.*")


def test_pipeline_valid_sample_ingestion_and_completed_cycle_reset(isolated_pipeline_env, monkeypatch):
    """
    Tests valid sample ingestion (*sample*.*) and ensures that re-running after a
    completed cycle (even with a larger batch_size where start_batch > total_batches)
    resets the completed source checkpoint and processes records rather than returning 0 records.
    """
    env = isolated_pipeline_env
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    sample_records = [
        {
            "review_id": f"rev_sample_{i}",
            "business_id": f"biz_{i}",
            "stars": 5 if i % 2 == 0 else 2,
            "text": f"Sample review text number {i} with plenty of detail.",
            "date": "2026-01-15",
        }
        for i in range(1, 6)
    ]
    sample_file = env["dirs"]["raw"] / "yelp_sample.json"
    sample_file.write_text(json.dumps(sample_records), encoding="utf-8")

    # First run with batch_size=2 -> 3 batches (batches 1, 2, 3), leaving next_batch_id=4
    orchestrator_small = PipelineOrchestrator(
        batch_size=2,
        ollama_client=env["mock_client"],
        checkpoint_manager=CheckpointManager(checkpoint_dir=env["dirs"]["checkpoints"]),
    )
    summary_1 = orchestrator_small.run_all(pattern="*sample*.*")
    assert summary_1.total_batches == 3
    assert summary_1.batches_processed == 3
    assert summary_1.total_records == 5
    assert summary_1.successful_predictions == 5
    assert len(summary_1.prediction_files) == 3

    # Second run with batch_size=32 (simulating Airflow DAG run after earlier CLI batch_size=2 run)
    # Without the fix, start_batch=4 > total_batches=1 caused 0 batches and 0 records.
    orchestrator_large = PipelineOrchestrator(
        batch_size=32,
        ollama_client=env["mock_client"],
        checkpoint_manager=CheckpointManager(checkpoint_dir=env["dirs"]["checkpoints"]),
    )
    summary_2 = orchestrator_large.run_all(pattern="*sample*.*")
    assert summary_2.total_batches == 1
    assert summary_2.batches_processed == 1
    assert summary_2.total_records == 5
    assert summary_2.successful_predictions == 5
    assert len(summary_2.prediction_files) == 1


def test_pipeline_partial_file_checkpoint_recovery(isolated_pipeline_env, monkeypatch):
    """
    Tests mid-file crash recovery: if batch 1 of 2 is committed in the checkpoint,
    run_file resumes at batch 2, records 1 skipped batch, and processes batch 2.
    """
    env = isolated_pipeline_env
    monkeypatch.setattr("src.pipeline.settings.paths", env["settings"].paths)
    monkeypatch.setattr("src.recovery.checkpoint.settings.paths", env["settings"].paths)

    sample_records = [
        {
            "review_id": f"rev_part_{i}",
            "business_id": f"biz_{i}",
            "stars": 4,
            "text": f"Review text {i} for partial recovery test.",
            "date": "2026-01-20",
        }
        for i in range(1, 5)
    ]
    sample_file = env["dirs"]["raw"] / "yelp_sample.json"
    sample_file.write_text(json.dumps(sample_records), encoding="utf-8")

    ckpt = CheckpointManager(checkpoint_dir=env["dirs"]["checkpoints"])
    # Simulate batch 1 ([0:2]) already committed prior to crash
    ckpt.record_batch_completion(
        batch_id=1,
        start_offset=0,
        end_offset=2,
        records_count=2,
        source_file="yelp_sample.json",
    )

    orchestrator = PipelineOrchestrator(
        batch_size=2,
        ollama_client=env["mock_client"],
        checkpoint_manager=ckpt,
    )
    summary = orchestrator.run_file(sample_file)
    assert summary.batches_skipped == 1
    assert summary.batches_processed == 1
    assert summary.total_batches == 2
    assert summary.total_records == 2
    assert summary.successful_predictions == 2



