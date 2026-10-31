"""Airflow DAG orchestrating batch runs of the Self-Healing Sentiment Pipeline with Ollama."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd

# Ensure project root is available in PYTHONPATH for Airflow workers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.utils.trigger_rule import TriggerRule
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_check_ollama_health(**context) -> bool:
    """Verifies that the Ollama LLM endpoint is reachable and responsive."""
    from src.inference.ollama_client import OllamaClient

    client = OllamaClient()
    is_healthy = client.is_healthy()
    if not is_healthy:
        raise ConnectionError(f"Ollama server unreachable at {client.base_url}")
    return True


def task_run_sentiment_pipeline(**context) -> Dict[str, Any]:
    """Executes the full self-healing sentiment analysis pipeline on configured raw datasets."""
    from src.pipeline import PipelineOrchestrator
    from src.utils.config import settings
    from src.utils.logger import get_logger

    logger = get_logger("airflow.run_pipeline")

    # Check for execution configuration passed via dag_run.conf
    dag_run = context.get("dag_run")
    conf = (dag_run.conf or {}) if dag_run else {}

    # Default to sample datasets for automated DAG runs to prevent unbound execution
    # of multi-gigabyte files (e.g. 5.34GB yelp_academic_dataset_review.json)
    pattern = conf.get("pattern") or conf.get("file_pattern")
    if not pattern:
        pattern = "*sample*.*"

    batch_size = conf.get("batch_size") or settings.ingestion.batch_size
    resolved_raw_dir = Path(settings.paths.raw_data_dir).resolve()

    logger.info(
        "Airflow task_run_sentiment_pipeline starting | resolved_raw_data_dir=%s | pattern=%s | batch_size=%d",
        resolved_raw_dir,
        pattern,
        batch_size,
    )

    orchestrator = PipelineOrchestrator(batch_size=batch_size)
    summary = orchestrator.run_all(pattern=pattern)

    prediction_output_paths = [str(Path(p).resolve()) for p in summary.prediction_files]
    logger.info(
        "Airflow task_run_sentiment_pipeline completed | resolved_raw_data_dir=%s | pattern=%s | "
        "batches_processed=%d | total_records=%d | prediction_output_paths=%s",
        resolved_raw_dir,
        pattern,
        summary.batches_processed,
        summary.total_records,
        prediction_output_paths,
    )

    if summary.total_records == 0 or summary.batches_processed == 0 or not prediction_output_paths:
        raise RuntimeError(
            f"Pipeline run failed non-empty validation for pattern='{pattern}' in '{resolved_raw_dir}': "
            f"batches_processed={summary.batches_processed}, total_records={summary.total_records}, "
            f"prediction_files_count={len(prediction_output_paths)}"
        )

    metrics = {
        "pattern": pattern,
        "resolved_raw_data_dir": str(resolved_raw_dir),
        "total_batches": summary.total_batches,
        "batches_processed": summary.batches_processed,
        "batches_skipped": summary.batches_skipped,
        "total_records": summary.total_records,
        "valid_records": summary.valid_records,
        "invalid_records": summary.invalid_records,
        "duplicate_records": summary.duplicate_records,
        "successful_predictions": summary.successful_predictions,
        "failed_predictions": summary.failed_predictions,
        "prediction_files_count": len(prediction_output_paths),
        "prediction_files": prediction_output_paths,
    }
    return metrics


def task_verify_pipeline_outputs(**context) -> Dict[str, Any]:
    """Verifies prediction persistence, atomic checkpoint state, and pipeline execution metrics."""
    from src.utils.config import settings
    from src.utils.logger import get_logger

    logger = get_logger("airflow.verify_outputs")
    logger.info("Executing output and checkpoint verification")

    # 1. Pull and validate execution metrics from upstream run_sentiment_pipeline task
    ti = context.get("task_instance")
    upstream_metrics = ti.xcom_pull(task_ids="run_sentiment_pipeline") if ti else None

    if upstream_metrics:
        logger.info("Upstream pipeline metrics retrieved via XCom: %s", upstream_metrics)
        total_batches = upstream_metrics.get("total_batches", 0)
        processed = upstream_metrics.get("batches_processed", 0)
        skipped = upstream_metrics.get("batches_skipped", 0)
        if total_batches == 0 and (processed + skipped) == 0:
            raise ValueError(
                f"Pipeline executed but evaluated 0 batches for pattern '{upstream_metrics.get('pattern')}'"
            )

    # 2. Validate prediction files in output/predictions
    pred_dir = Path(settings.paths.predictions_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Predictions directory does not exist: {pred_dir}")

    pred_files = [f for f in pred_dir.glob("*.parquet") if f.name != ".gitkeep"]
    json_pred_files = [f for f in pred_dir.glob("*.json") if f.name != ".gitkeep"]
    all_pred_files = pred_files + json_pred_files

    if not all_pred_files:
        raise FileNotFoundError(
            f"No prediction files (.parquet or .json) found in {pred_dir}"
        )

    # Validate schema of the latest prediction file
    latest_pred_file = max(all_pred_files, key=lambda f: f.stat().st_mtime)
    if latest_pred_file.suffix == ".parquet":
        pred_df = pd.read_parquet(latest_pred_file)
    else:
        pred_df = pd.read_json(latest_pred_file)

    if pred_df.empty:
        raise ValueError(f"Latest prediction file {latest_pred_file.name} is empty (0 records)")

    required_cols = {"review_id", "sentiment", "confidence"}
    missing_cols = required_cols - set(pred_df.columns)
    if missing_cols:
        raise ValueError(
            f"Prediction file {latest_pred_file.name} is missing required columns: {missing_cols}"
        )

    logger.info(
        "Validated %d prediction files in %s. Latest file: %s (%d records)",
        len(all_pred_files),
        pred_dir,
        latest_pred_file.name,
        len(pred_df),
    )

    # 3. Validate checkpoint state in checkpoints/pipeline_checkpoint.json
    checkpoint_file = Path(settings.paths.checkpoint_dir) / "pipeline_checkpoint.json"
    if not checkpoint_file.exists() or checkpoint_file.stat().st_size == 0:
        raise FileNotFoundError(
            f"Pipeline checkpoint file missing or empty: {checkpoint_file}"
        )

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        checkpoint_data = json.load(f)

    completed_batches = checkpoint_data.get("completed_batches", {})
    last_completed = checkpoint_data.get("last_completed_batch", 0)
    last_batch_id = last_completed.get("batch_id", 0) if isinstance(last_completed, dict) else (last_completed or 0)

    if not completed_batches or last_batch_id <= 0:
        raise ValueError(
            f"Checkpoint {checkpoint_file.name} contains no completed batches (last_batch_id={last_batch_id})"
        )

    logger.info(
        "Validated checkpoint state: %d completed batches recorded (last_completed_batch=%d)",
        len(completed_batches),
        last_batch_id,
    )

    verification_summary = {
        "status": "VERIFIED",
        "total_prediction_files": len(all_pred_files),
        "latest_prediction_file": latest_pred_file.name,
        "latest_prediction_records": len(pred_df),
        "checkpoint_completed_batches": len(completed_batches),
        "checkpoint_last_batch": last_batch_id,
        "upstream_metrics": upstream_metrics,
    }
    return verification_summary


default_args = {
    "owner": "data_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
}

if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id="sentiment_pipeline_dag",
        default_args=default_args,
        description="Daily execution of Self-Healing Sentiment Analysis Pipeline with Ollama",
        schedule_interval="@daily",
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["sentiment", "llm", "self-healing", "ollama"],
    ) as dag:
        check_health = PythonOperator(
            task_id="check_ollama_health",
            python_callable=task_check_ollama_health,
        )

        execute_pipeline = PythonOperator(
            task_id="run_sentiment_pipeline",
            python_callable=task_run_sentiment_pipeline,
        )

        verify_outputs = PythonOperator(
            task_id="verify_pipeline_outputs",
            python_callable=task_verify_pipeline_outputs,
            trigger_rule=TriggerRule.ALL_SUCCESS,
        )

        check_health >> execute_pipeline >> verify_outputs
