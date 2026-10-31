"""Ollama client integration with connection health checks, timeouts, and error handling."""

from typing import Any, Dict, List, Optional, Union
import httpx
import ollama

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("inference.ollama_client")


class OllamaClientError(Exception):
    """Base exception for all Ollama client failures."""


class OllamaConnectionError(OllamaClientError):
    """Raised when unable to establish connection to the Ollama server."""


class OllamaTimeoutError(OllamaClientError):
    """Raised when an inference or health check request times out."""


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the requested model is not pulled or available in Ollama."""


class OllamaResponseError(OllamaClientError):
    """Raised when Ollama returns an error response during generation."""


class OllamaClient:
    """Wrapper around Ollama client providing connection resilience and structured inference."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.ollama.base_url).rstrip("/")
        self.model = model or settings.ollama.model
        self.timeout = timeout or settings.ollama.timeout_seconds
        self._client = ollama.Client(host=self.base_url, timeout=self.timeout)

    def is_healthy(self) -> bool:
        """Verifies if the Ollama server is reachable and responsive."""
        try:
            self._client.list()
            return True
        except (httpx.ConnectError, httpx.NetworkError, ConnectionError):
            logger.warning("Ollama server unreachable at %s", self.base_url)
            return False
        except httpx.TimeoutException:
            logger.warning("Ollama server health check timed out at %s", self.base_url)
            return False
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """Checks if a specified model (defaulting to configured model) is installed locally."""
        target_model = (model_name or self.model).strip()
        try:
            response = self._client.list()
            # ollama.list() returns ListResponse with models attribute containing Model objects
            models = getattr(response, "models", [])
            installed_names = []
            for m in models:
                name = getattr(m, "model", None) or getattr(m, "name", None) or str(m)
                installed_names.append(name.lower())

            # Check exact match or prefix before ':latest'
            target_lower = target_model.lower()
            target_base = target_lower.split(":")[0]

            for installed in installed_names:
                if installed == target_lower or installed == f"{target_lower}:latest":
                    return True
                if installed.startswith(f"{target_lower}:") or installed.split(":")[0] == target_base:
                    return True

            logger.info("Model '%s' not found among installed models: %s", target_model, installed_names)
            return False
        except Exception as exc:
            self._map_and_raise_error(exc)
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        format: Optional[Union[str, Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Sends a chat completion request to the configured model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'.
            format: Optional format constraint ('json' or JSON schema dictionary).
            options: Optional runtime options (temperature, top_p, etc.).

        Returns:
            str: Raw text content from the assistant's reply.
        """
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if format is not None:
                kwargs["format"] = format
            if options is not None:
                kwargs["options"] = options

            response = self._client.chat(**kwargs)
            message = getattr(response, "message", None)
            if message and hasattr(message, "content"):
                return message.content
            if isinstance(response, dict) and "message" in response:
                return response["message"].get("content", "")
            return str(response)

        except Exception as exc:
            self._map_and_raise_error(exc)
            return ""

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        format: Optional[Union[str, Dict[str, Any]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Sends a completion request to the configured model."""
        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "prompt": prompt,
            }
            if system:
                kwargs["system"] = system
            if format is not None:
                kwargs["format"] = format
            if options is not None:
                kwargs["options"] = options

            response = self._client.generate(**kwargs)
            if hasattr(response, "response"):
                return response.response
            if isinstance(response, dict):
                return response.get("response", "")
            return str(response)

        except Exception as exc:
            self._map_and_raise_error(exc)
            return ""

    def _map_and_raise_error(self, exc: Exception) -> None:
        """Maps third-party HTTP and Ollama client exceptions to domain exceptions."""
        if isinstance(exc, (OllamaClientError,)):
            raise exc

        if isinstance(exc, (httpx.ConnectError, ConnectionRefusedError, ConnectionError)):
            logger.error("Connection error reaching Ollama at %s: %s", self.base_url, exc)
            raise OllamaConnectionError(f"Could not connect to Ollama at {self.base_url}: {exc}") from exc

        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            logger.error("Ollama request timed out after %s seconds: %s", self.timeout, exc)
            raise OllamaTimeoutError(f"Ollama request timed out after {self.timeout}s: {exc}") from exc

        if isinstance(exc, ollama.ResponseError):
            if exc.status_code == 404 or "not found" in str(exc.error).lower():
                logger.error("Model '%s' not found on Ollama server: %s", self.model, exc.error)
                raise OllamaModelNotFoundError(f"Model '{self.model}' not found in Ollama: {exc.error}") from exc
            logger.error("Ollama responded with error code %s: %s", exc.status_code, exc.error)
            raise OllamaResponseError(f"Ollama error ({exc.status_code}): {exc.error}") from exc

        logger.error("Unexpected error in Ollama client: %s", exc)
        raise OllamaClientError(f"Ollama invocation failed: {exc}") from exc

