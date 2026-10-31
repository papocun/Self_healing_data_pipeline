# Self-Healing Sentiment Analysis Pipeline

An automated data pipeline with self-healing, retry mechanisms, and checkpointing for sentiment analysis powered by Ollama LLM and monitored via a Streamlit dashboard.

## Project Structure

```text
SelfHealingPipeline/
│
├── dags/
│   └── sentiment_pipeline_dag.py
│
├── src/
│   ├── ingestion/
│   │   ├── batch_loader.py
│   │   └── data_reader.py
│   │
│   ├── processing/
│   │   ├── validator.py
│   │   ├── cleaner.py
│   │   └── deduplicator.py
│   │
│   ├── inference/
│   │   ├── ollama_client.py
│   │   └── sentiment_analyzer.py
│   │
│   ├── recovery/
│   │   ├── checkpoint.py
│   │   └── retry_handler.py
│   │
│   ├── analytics/
│   │   └── aggregator.py
│   │
│   └── utils/
│       ├── config.py
│       └── logger.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── failed/
│
├── output/
│   ├── predictions/
│   └── analytics/
│
├── dashboard/
│   ├── app.py
│   ├── components/
│   │   ├── metrics.py
│   │   ├── charts.py
│   │   └── filters.py
│   └── pages/
│       ├── 01_Sentiment_Overview.py
│       ├── 02_Review_Explorer.py
│       └── 03_Pipeline_Monitor.py
│
├── tests/
│   ├── __init__.py
│   └── conftest.py
├── checkpoints/
├── logs/
│
├── Dockerfile
├── docker-compose.yml
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup Instructions

1. **Clone the repository and enter the directory:**
   ```bash
   cd SelfHealingPipeline
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   ```

5. **Run the pipeline:**
   ```bash
   python main.py
   ```

6. **Launch the Dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

