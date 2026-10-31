"""Pipeline entry point supporting full batch execution and live inference verification."""

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

from src.inference.ollama_client import OllamaClient, OllamaConnectionError
from src.inference.sentiment_analyzer import SentimentAnalyzer
from src.pipeline import PipelineOrchestrator
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("main")


def run_live_inference_check(review_id: str, review_text: str) -> None:
    """Executes a live single-review inference check against local Ollama."""
    print("=" * 60)
    print("Self-Healing Pipeline: Live Inference Check")
    print("=" * 60)
    print(f"Target API Endpoint: {settings.ollama.base_url}")
    print(f"Target Model:        {settings.ollama.model}")
    print(f"Request Timeout:     {settings.ollama.timeout_seconds}s")
    print("-" * 60)

    client = OllamaClient()

    print("[1/3] Checking Ollama connectivity...")
    if not client.is_healthy():
        print(f"❌ ERROR: Cannot connect to Ollama at {settings.ollama.base_url}.")
        print("   Please ensure Ollama is running (`ollama serve`).")
        sys.exit(1)
    print("✔ Ollama server is reachable.")

    print(f"[2/3] Verifying availability of model '{client.model}'...")
    try:
        if not client.is_model_available():
            print(f"❌ Model '{client.model}' is not pulled.")
            print(f"   Please pull the model first by running:\n   ollama pull {client.model}")
            sys.exit(1)
        print(f"✔ Model '{client.model}' is ready.")
    except OllamaConnectionError as exc:
        print(f"❌ Connection error: {exc}")
        sys.exit(1)

    print("[3/3] Running sentiment analysis with structured output...")
    analyzer = SentimentAnalyzer(client=client)
    result = analyzer.analyze_review(review_id=review_id, text=review_text)

    print("\n" + "=" * 60)
    print("Inference Result (Validated Schema):")
    print("=" * 60)
    print(json.dumps(result.model_dump(), indent=2))
    print("=" * 60)


def run_pipeline(file_path: str = None, batch_size: int = None) -> None:
    """Executes the end-to-end self-healing pipeline."""
    print("=" * 60)
    print("Starting Self-Healing Sentiment Analysis Pipeline")
    print("=" * 60)
    print(f"Model:      {settings.ollama.model}")
    print(f"Batch Size: {batch_size or settings.ingestion.batch_size}")
    print(f"Raw Dir:    {settings.paths.raw_data_dir}")
    print(f"Output Dir: {settings.paths.predictions_dir}")
    print("-" * 60)

    orchestrator = PipelineOrchestrator(batch_size=batch_size)

    if file_path:
        target = Path(file_path)
        if not target.exists():
            print(f"❌ ERROR: Specified input file not found: {target}")
            sys.exit(1)
        summary = orchestrator.run_file(target)
    else:
        summary = orchestrator.run_all()

    print("\n" + "=" * 60)
    print("Pipeline Execution Summary:")
    print("=" * 60)
    print(f"Total Batches Scanned:   {summary.total_batches}")
    print(f"Batches Processed:       {summary.batches_processed}")
    print(f"Batches Skipped (Prior): {summary.batches_skipped}")
    print(f"Total Records Ingested:  {summary.total_records}")
    print(f"Valid Records:           {summary.valid_records}")
    print(f"Invalid Records (Fail):  {summary.invalid_records}")
    print(f"Duplicates Quarantined:  {summary.duplicate_records}")
    print(f"Predictions Written:     {summary.successful_predictions}")
    print(f"Failed Inference Items:  {summary.failed_predictions}")
    print(f"Prediction Files:        {len(summary.prediction_files)}")
    print(f"Quarantine Files:        {len(summary.quarantine_files)}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Healing Sentiment Analysis Pipeline CLI")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Optional path to a specific raw dataset file (CSV/JSON/JSONL/Parquet)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override default batch size",
    )
    parser.add_argument(
        "--test-inference",
        action="store_true",
        help="Run a live single-review sentiment inference check against local Ollama",
    )
    parser.add_argument(
        "--review-id",
        type=str,
        default="rev_live_001",
        help="Review ID for live inference test",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The battery life is exceptional and the display is crystal clear, but customer support was unhelpful.",
        help="Review text for live inference test",
    )

    args = parser.parse_args()

    if args.test_inference:
        run_live_inference_check(review_id=args.review_id, review_text=args.text)
    else:
        run_pipeline(file_path=args.file, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
