"""Unit tests for OllamaClient and SentimentAnalyzer using mocked Ollama responses."""

import json
from unittest.mock import MagicMock, patch
import httpx
import ollama
import pytest

from src.inference.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from src.inference.sentiment_analyzer import (
    MalformedModelResponseError,
    SentimentAnalysisError,
    SentimentAnalyzer,
    SentimentResult,
)


# ============================================================================
# OllamaClient Tests
# ============================================================================


def test_ollama_client_init():
    client = OllamaClient(base_url="http://test-server:11434", model="qwen2.5:3b", timeout=45.0)
    assert client.base_url == "http://test-server:11434"
    assert client.model == "qwen2.5:3b"
    assert client.timeout == 45.0


def test_ollama_client_is_healthy_success():
    client = OllamaClient()
    client._client = MagicMock()
    client._client.list.return_value = MagicMock()

    assert client.is_healthy() is True
    client._client.list.assert_called_once()


def test_ollama_client_is_healthy_unreachable():
    client = OllamaClient()
    client._client = MagicMock()
    client._client.list.side_effect = httpx.ConnectError("Failed to connect")

    assert client.is_healthy() is False


def test_ollama_client_is_healthy_timeout():
    client = OllamaClient()
    client._client = MagicMock()
    client._client.list.side_effect = httpx.TimeoutException("Read timed out")

    assert client.is_healthy() is False


def test_ollama_client_is_model_available():
    client = OllamaClient(model="qwen2.5:3b")
    client._client = MagicMock()

    # Mock list response containing models
    mock_model_1 = MagicMock()
    mock_model_1.model = "qwen2.5:3b"
    mock_model_2 = MagicMock()
    mock_model_2.model = "llama3:latest"

    list_resp = MagicMock()
    list_resp.models = [mock_model_1, mock_model_2]
    client._client.list.return_value = list_resp

    assert client.is_model_available("qwen2.5:3b") is True
    assert client.is_model_available("llama3") is True
    assert client.is_model_available("mistral:7b") is False


def test_ollama_client_chat_success():
    client = OllamaClient(model="qwen2.5:3b")
    client._client = MagicMock()

    mock_chat_resp = MagicMock()
    mock_chat_resp.message.content = '{"sentiment": "positive"}'
    client._client.chat.return_value = mock_chat_resp

    result = client.chat(messages=[{"role": "user", "content": "hi"}], format="json")
    assert result == '{"sentiment": "positive"}'
    client._client.chat.assert_called_once_with(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": "hi"}],
        format="json",
    )


def test_ollama_client_chat_connection_error():
    client = OllamaClient()
    client._client = MagicMock()
    client._client.chat.side_effect = httpx.ConnectError("Connection refused")

    with pytest.raises(OllamaConnectionError):
        client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_chat_timeout_error():
    client = OllamaClient()
    client._client = MagicMock()
    client._client.chat.side_effect = httpx.TimeoutException("Operation timed out")

    with pytest.raises(OllamaTimeoutError):
        client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_chat_model_not_found():
    client = OllamaClient(model="nonexistent_model")
    client._client = MagicMock()
    client._client.chat.side_effect = ollama.ResponseError("model 'nonexistent_model' not found", status_code=404)

    with pytest.raises(OllamaModelNotFoundError):
        client.chat(messages=[{"role": "user", "content": "hi"}])


def test_ollama_client_generate_success():
    client = OllamaClient(model="qwen2.5:3b")
    client._client = MagicMock()

    mock_gen_resp = MagicMock()
    mock_gen_resp.response = "Generated answer"
    client._client.generate.return_value = mock_gen_resp

    result = client.generate(prompt="Explain tests", system="Be concise")
    assert result == "Generated answer"
    client._client.generate.assert_called_once_with(
        model="qwen2.5:3b",
        prompt="Explain tests",
        system="Be concise",
    )


# ============================================================================
# SentimentAnalyzer Tests
# ============================================================================


@pytest.fixture
def mock_ollama_client():
    return MagicMock(spec=OllamaClient)


def test_analyze_empty_review_returns_neutral():
    analyzer = SentimentAnalyzer(client=MagicMock())
    result = analyzer.analyze_review("rev_001", "   ")

    assert isinstance(result, SentimentResult)
    assert result.review_id == "rev_001"
    assert result.sentiment == "neutral"
    assert result.confidence == 1.0
    assert result.positive_aspects == []
    assert result.negative_aspects == []


def test_analyze_positive_review(mock_ollama_client):
    payload = {
        "sentiment": "positive",
        "confidence": 0.98,
        "positive_aspects": ["build quality", "battery life"],
        "negative_aspects": [],
        "reasoning": "User expressed satisfaction with hardware durability.",
    }
    mock_ollama_client.chat.return_value = json.dumps(payload)

    analyzer = SentimentAnalyzer(client=mock_ollama_client)
    res = analyzer.analyze_review("rev_100", "The build quality and battery life are phenomenal!")

    assert res.review_id == "rev_100"
    assert res.sentiment == "positive"
    assert res.confidence == 0.98
    assert res.positive_aspects == ["build quality", "battery life"]
    assert res.negative_aspects == []
    assert res.reasoning == "User expressed satisfaction with hardware durability."


def test_analyze_negative_review(mock_ollama_client):
    payload = {
        "sentiment": "negative",
        "confidence": 0.95,
        "positive_aspects": [],
        "negative_aspects": ["shipping delay", "unhelpful support"],
        "reasoning": "Order arrived 2 weeks late with poor customer service.",
    }
    mock_ollama_client.chat.return_value = json.dumps(payload)

    analyzer = SentimentAnalyzer(client=mock_ollama_client)
    res = analyzer.analyze_review("rev_101", "Support was unhelpful and shipping was delayed.")

    assert res.sentiment == "negative"
    assert res.confidence == 0.95
    assert "shipping delay" in res.negative_aspects


def test_analyze_normalizes_casing_and_markdown_wrappers(mock_ollama_client):
    raw_markdown_output = """```json
    {
        "sentiment": "POSITIVE",
        "confidence": 0.92,
        "positive_aspects": ["screen"],
        "negative_aspects": null
    }
    ```"""
    mock_ollama_client.chat.return_value = raw_markdown_output

    analyzer = SentimentAnalyzer(client=mock_ollama_client)
    res = analyzer.analyze_review("rev_102", "Love the screen.")

    assert res.sentiment == "positive"
    assert res.confidence == 0.92
    assert res.positive_aspects == ["screen"]
    assert res.negative_aspects == []


def test_analyze_fallback_substring_extraction(mock_ollama_client):
    raw_conversational_output = (
        'Here is the analysis:\n{"sentiment": "neutral", "confidence": 0.85, '
        '"positive_aspects": [], "negative_aspects": []}\nHope that helps!'
    )
    mock_ollama_client.chat.return_value = raw_conversational_output

    analyzer = SentimentAnalyzer(client=mock_ollama_client)
    res = analyzer.analyze_review("rev_103", "It is just okay.")

    assert res.sentiment == "neutral"
    assert res.confidence == 0.85


def test_analyze_invalid_sentiment_raises_malformed_error(mock_ollama_client):
    mock_ollama_client.chat.return_value = json.dumps(
        {"sentiment": "mixed", "confidence": 0.5, "positive_aspects": [], "negative_aspects": []}
    )

    analyzer = SentimentAnalyzer(client=mock_ollama_client)
    with pytest.raises(MalformedModelResponseError):
        analyzer.analyze_review("rev_104", "Mixed feelings.")


def test_analyze_non_json_raises_malformed_error(mock_ollama_client):
    mock_ollama_client.chat.return_value = "Sorry, I cannot analyze this text."

    analyzer = SentimentAnalyzer(client=mock_ollama_client)
    with pytest.raises(MalformedModelResponseError):
        analyzer.analyze_review("rev_105", "Some review text")


def test_analyze_wraps_client_error(mock_ollama_client):
    mock_ollama_client.chat.side_effect = OllamaConnectionError("Cannot reach server")

    analyzer = SentimentAnalyzer(client=mock_ollama_client)
    with pytest.raises(SentimentAnalysisError) as exc_info:
        analyzer.analyze_review("rev_106", "Review text")
    assert "Inference failed for review rev_106" in str(exc_info.value)


def test_analyze_batch(mock_ollama_client):
    mock_ollama_client.chat.side_effect = [
        json.dumps({"sentiment": "positive", "confidence": 0.99, "positive_aspects": ["fast"]}),
        json.dumps({"sentiment": "negative", "confidence": 0.91, "negative_aspects": ["slow"]}),
    ]

    analyzer = SentimentAnalyzer(client=mock_ollama_client)
    batch = [
        {"review_id": "r1", "text": "Very fast"},
        {"review_id": "r2", "text": "Extremely slow"},
    ]
    results = analyzer.analyze_batch(batch)

    assert len(results) == 2
    assert results[0].review_id == "r1"
    assert results[0].sentiment == "positive"
    assert results[1].review_id == "r2"
    assert results[1].sentiment == "negative"

