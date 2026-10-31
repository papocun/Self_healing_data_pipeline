"""Configurable retry execution handler with exponential backoff for transient failures."""

from functools import wraps
import time
from typing import Any, Callable, Optional, Sequence, Tuple, Type, TypeVar
import httpx

from src.inference.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from src.inference.sentiment_analyzer import MalformedModelResponseError
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger("recovery.retry_handler")

T = TypeVar("T")

# Transient errors that warrant retry attempts
DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    OllamaConnectionError,
    OllamaTimeoutError,
    httpx.ConnectError,
    httpx.TimeoutException,
    ConnectionError,
    TimeoutError,
)

# Deterministic or data-level errors that must fail fast without retry
DEFAULT_NON_RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    MalformedModelResponseError,
    OllamaModelNotFoundError,
    ValueError,
    TypeError,
    KeyError,
)


class RetryExhaustedError(Exception):
    """Raised when maximum retry attempts have been exhausted."""

    def __init__(self, message: str, attempts: int, last_exception: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_exception = last_exception


class RetryHandler:
    """Executes callables with configurable exponential backoff and error classification."""

    def __init__(
        self,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
    ):
        self.max_retries = (
            max_retries if max_retries is not None else settings.recovery.max_retries
        )
        self.initial_delay = (
            initial_delay if initial_delay is not None else settings.recovery.retry_delay_seconds
        )
        self.backoff_factor = (
            backoff_factor if backoff_factor is not None else settings.recovery.backoff_factor
        )
        self.retryable_exceptions = tuple(retryable_exceptions or DEFAULT_RETRYABLE_EXCEPTIONS)
        self.non_retryable_exceptions = tuple(
            non_retryable_exceptions or DEFAULT_NON_RETRYABLE_EXCEPTIONS
        )

    def is_retryable(self, exc: Exception) -> bool:
        """Determines if an exception is classified as a transient retryable failure."""
        if isinstance(exc, self.non_retryable_exceptions):
            return False
        return isinstance(exc, self.retryable_exceptions)

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Executes a callable with exponential backoff on retryable exceptions.

        Args:
            func: Target callable.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            T: Output of func on success.

        Raises:
            RetryExhaustedError: If retries are exhausted.
            Exception: Any non-retryable exception immediately.
        """
        attempt = 0
        current_delay = self.initial_delay
        func_name = getattr(func, "__name__", type(func).__name__)

        while True:
            attempt += 1
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                if not self.is_retryable(exc):
                    logger.warning(
                        "Non-retryable error encountered (%s: %s); failing fast without retry",
                        type(exc).__name__,
                        exc,
                    )
                    raise

                if attempt > self.max_retries:
                    error_msg = (
                        f"Retry exhausted for {func_name} after {attempt - 1} retries. "
                        f"Last error: {type(exc).__name__}: {exc}"
                    )
                    logger.error(error_msg)
                    raise RetryExhaustedError(
                        message=error_msg,
                        attempts=attempt - 1,
                        last_exception=exc,
                    ) from exc

                logger.warning(
                    "Transient failure in %s (attempt %d/%d): %s: %s. Retrying in %.2fs...",
                    func_name,
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    exc,
                    current_delay,
                )

                time.sleep(current_delay)
                current_delay *= self.backoff_factor


def retry_with_backoff(
    max_retries: Optional[int] = None,
    initial_delay: Optional[float] = None,
    backoff_factor: Optional[float] = None,
    retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
    non_retryable_exceptions: Optional[Sequence[Type[Exception]]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for wrapping functions with retry execution."""
    handler = RetryHandler(
        max_retries=max_retries,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return handler.execute(func, *args, **kwargs)

        return wrapper

    return decorator
