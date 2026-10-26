from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from schemas import ChatMessage, LLMConfig, ModelResponse


class BaseLLMClient(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    async def generate(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> ModelResponse:
        """Genera una respuesta completa de manera asíncrona."""

    @abstractmethod
    async def generate_stream(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> AsyncGenerator[str, None]:
        """Genera tokens de manera asíncrona vía streaming (yield)."""
