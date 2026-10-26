from collections.abc import AsyncGenerator

from base import BaseLLMClient
from clients import AnthropicClient, GeminiClient, OpenAIClient
from schemas import ChatMessage, LLMConfig, ModelResponse, Provider


class AsyncLLMManager:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = self._crear_cliente()

    def _crear_cliente(self) -> BaseLLMClient:
        if self.config.provider == Provider.OPENAI:
            if not self.config.openai_api_key:
                raise ValueError("Falta openai_api_key en la configuración")
            return OpenAIClient(
                api_key=self.config.openai_api_key.get_secret_value()
            )

        if self.config.provider == Provider.ANTHROPIC:
            if not self.config.anthropic_api_key:
                raise ValueError("Falta anthropic_api_key en la configuración")
            return AnthropicClient(
                api_key=self.config.anthropic_api_key.get_secret_value()
            )

        if self.config.provider == Provider.GEMINI:
            if not self.config.google_api_key:
                raise ValueError("Falta google_api_key en la configuración")
            return GeminiClient(
                api_key=self.config.google_api_key.get_secret_value()
            )

        raise ValueError(f"Proveedor no soportado: {self.config.provider}")

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        return await self._client.generate(messages, self.config)

    async def generate_stream(
        self, messages: list[ChatMessage]
    ) -> AsyncGenerator[str, None]:
        async for token in self._client.generate_stream(messages, self.config):
            yield token