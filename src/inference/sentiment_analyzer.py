"""Sentiment analysis engine using Ollama LLM with structured output schema enforcement."""

import json
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError

from src.inference.ollama_client import OllamaClient, OllamaClientError
from src.utils.logger import get_logger

logger = get_logger("inference.sentiment_analyzer")


class SentimentAnalysisError(Exception):
    """Base exception for sentiment analysis failures."""


class MalformedModelResponseError(SentimentAnalysisError):
    """Raised when the LLM produces output that cannot be parsed into the expected schema."""


class SentimentPayload(BaseModel):
    """Structured schema enforced on the Ollama generation."""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    positive_aspects: List[str] = Field(default_factory=list)
    negative_aspects: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None


class SentimentResult(BaseModel):
    """Validated final sentiment record associated with a review_id."""
    review_id: str
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    positive_aspects: List[str]
    negative_aspects: List[str]
    reasoning: Optional[str] = None
    raw_text: Optional[str] = None


_SYSTEM_PROMPT = """You are an expert customer feedback sentiment analyzer.
Your task is to analyze the review text and determine:
1. sentiment: Exactly one of "positive", "negative", or "neutral".
2. confidence: A float between 0.0 and 1.0 indicating classification confidence.
3. positive_aspects: List of distinct positive features, qualities, or aspects praised.
4. negative_aspects: List of distinct negative features, qualities, or pain points mentioned.
5. reasoning: A brief explanation of the sentiment classification.

You must respond ONLY with valid JSON strictly conforming to the requested schema."""


class SentimentAnalyzer:
    """Analyzes customer review sentiment using Ollama with JSON schema structured output."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        self.schema = SentimentPayload.model_json_schema()

    def analyze_review(self, review_id: str, text: str) -> SentimentResult:
        """Classifies sentiment and extracts aspects for a single review.

        Args:
            review_id: Unique identifier for the review.
            text: Raw customer review text.

        Returns:
            SentimentResult: Validated structured sentiment result.

        Raises:
            SentimentAnalysisError: If the model call fails or output is unrecoverably malformed.
        """
        cleaned_text = (text or "").strip()

        # Handle empty/whitespace-only input deterministically without invoking the LLM
        if not cleaned_text:
            return SentimentResult(
                review_id=str(review_id),
                sentiment="neutral",
                confidence=1.0,
                positive_aspects=[],
                negative_aspects=[],
                reasoning="Empty review text provided.",
                raw_text="",
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Review Text:\n\"\"\"{cleaned_text}\"\"\""},
        ]

        try:
            raw_response = self.client.chat(
                messages=messages,
                format=self.schema,
                options={"temperature": 0.0},  # Deterministic output
            )
        except OllamaClientError as exc:
            logger.error("Inference failure for review %s: %s", review_id, exc)
            raise SentimentAnalysisError(f"Inference failed for review {review_id}: {exc}") from exc

        parsed_payload = self._parse_and_validate(raw_response, review_id)

        return SentimentResult(
            review_id=str(review_id),
            sentiment=parsed_payload.sentiment,
            confidence=round(parsed_payload.confidence, 4),
            positive_aspects=parsed_payload.positive_aspects,
            negative_aspects=parsed_payload.negative_aspects,
            reasoning=parsed_payload.reasoning,
            raw_text=cleaned_text,
        )

    def analyze_batch(self, reviews: List[Dict[str, Any]]) -> List[SentimentResult]:
        """Analyzes a collection of review dictionaries containing 'review_id' and 'text'."""
        results: List[SentimentResult] = []
        for item in reviews:
            rev_id = str(item.get("review_id", ""))
            text = str(item.get("text", ""))
            results.append(self.analyze_review(rev_id, text))
        return results

    def _parse_and_validate(self, raw_output: str, review_id: str) -> SentimentPayload:
        """Parses model response, attempting cleanup if the model returned markdown codeblocks."""
        if not raw_output or not raw_output.strip():
            raise MalformedModelResponseError(f"Model returned empty response for review {review_id}")

        cleaned = raw_output.strip()

        # Strip markdown ```json ... ``` wrapper if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Fast path: direct JSON parse
        try:
            data = json.loads(cleaned)
            return self._validate_payload_dict(data, review_id)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Fallback: extract substring between first '{' and last '}'
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                data = json.loads(snippet)
                return self._validate_payload_dict(data, review_id)
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.error("JSON snippet validation failed for review %s: %s", review_id, exc)

        raise MalformedModelResponseError(
            f"Unable to parse structured sentiment response for review {review_id}. Raw output: {raw_output[:200]}"
        )

    def _validate_payload_dict(self, data: Dict[str, Any], review_id: str) -> SentimentPayload:
        """Normalizes and validates raw dict against SentimentPayload."""
        if not isinstance(data, dict):
            raise MalformedModelResponseError(f"Expected JSON object, got {type(data).__name__}")

        # Normalize sentiment label casing
        if "sentiment" in data and isinstance(data["sentiment"], str):
            norm_sentiment = data["sentiment"].strip().lower()
            if norm_sentiment in ("positive", "pos"):
                data["sentiment"] = "positive"
            elif norm_sentiment in ("negative", "neg"):
                data["sentiment"] = "negative"
            elif norm_sentiment in ("neutral", "neu"):
                data["sentiment"] = "neutral"

        # Ensure aspect lists are valid lists of strings
        for aspect_key in ("positive_aspects", "negative_aspects"):
            val = data.get(aspect_key)
            if val is None:
                data[aspect_key] = []
            elif isinstance(val, str):
                data[aspect_key] = [val.strip()] if val.strip() else []
            elif isinstance(val, list):
                data[aspect_key] = [str(x).strip() for x in val if str(x).strip()]

        try:
            return SentimentPayload.model_validate(data)
        except ValidationError as val_err:
            raise MalformedModelResponseError(
                f"Validation error for review {review_id}: {val_err}"
            ) from val_err

