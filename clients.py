import os
from collections.abc import AsyncGenerator

import anthropic
import openai
from google import genai
from google.genai import types

from base import BaseLLMClient
from schemas import ChatMessage, LLMConfig, ModelResponse, Provider


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = openai.AsyncOpenAI(api_key=self.api_key)

    async def generate(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> ModelResponse:
        try:
            formatted_messages = [
                {"role": m.role, "content": m.content} for m in messages
            ]
            response = await self.client.chat.completions.create(
                model=config.model,
                messages=formatted_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            content = response.choices[0].message.content or ""
            return ModelResponse(
                provider=Provider.OPENAI,
                model=config.model,
                content=content,
            )
        except Exception as e:  # noqa: BLE001
            return ModelResponse(
                provider=Provider.OPENAI,
                model=config.model,
                content="",
                error=str(e),
            )

    async def generate_stream(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> AsyncGenerator[str, None]:
        formatted_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        stream = await self.client.chat.completions.create(
            model=config.model,
            messages=formatted_messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, workspace_id: str | None = None):
        super().__init__(api_key)
        ws_id = workspace_id or os.getenv("ANTHROPIC_WORKSPACE_ID")
        headers = {}
        if ws_id:
            headers["anthropic-workspace-id"] = ws_id

        self.client = anthropic.AsyncAnthropic(
            api_key=self.api_key,
            default_headers=headers if headers else None,
        )

    def _extract_system_and_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict]]:
        system_prompt = None
        formatted_messages = []
        for m in messages:
            if m.role == "system":
                system_prompt = m.content
            else:
                formatted_messages.append({"role": m.role, "content": m.content})
        return system_prompt, formatted_messages

    async def generate(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> ModelResponse:
        system_prompt, formatted_messages = self._extract_system_and_messages(
            messages
        )
        params = {
            "model": config.model,
            "messages": formatted_messages,
            "max_tokens": config.max_tokens,
        }
        if system_prompt:
            params["system"] = system_prompt

        try:
            response = await self.client.messages.create(**params)
            content = response.content[0].text if response.content else ""
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=config.model,
                content=content,
            )
        except Exception as e:  # noqa: BLE001
            return ModelResponse(
                provider=Provider.ANTHROPIC,
                model=config.model,
                content="",
                error=str(e),
            )

    async def generate_stream(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> AsyncGenerator[str, None]:
        system_prompt, formatted_messages = self._extract_system_and_messages(
            messages
        )
        params = {
            "model": config.model,
            "messages": formatted_messages,
            "max_tokens": config.max_tokens,
        }
        if system_prompt:
            params["system"] = system_prompt

        try:
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:  # noqa: BLE001
            yield f"Error en Anthropic Stream: {e}"


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = genai.Client(api_key=self.api_key)

    async def generate(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> ModelResponse:
        try:
            prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])
            gen_config = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )
            # Usamos AsyncChat para eliminar la advertencia de Automatic Function Calling (AFC)
            chat = self.client.aio.chats.create(model=config.model, config=gen_config)
            response = await chat.send_message(prompt)
            return ModelResponse(
                provider=Provider.GEMINI,
                model=config.model,
                content=response.text or "",
            )
        except Exception as e:  # noqa: BLE001
            return ModelResponse(
                provider=Provider.GEMINI,
                model=config.model,
                content="",
                error=str(e),
            )

    async def generate_stream(
        self, messages: list[ChatMessage], config: LLMConfig
    ) -> AsyncGenerator[str, None]:
        try:
            prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])
            gen_config = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )
            chat = self.client.aio.chats.create(model=config.model, config=gen_config)
            async for chunk in await chat.send_message_stream(prompt):
                if chunk.text:
                    yield chunk.text
        except Exception as e:  # noqa: BLE001
            yield f"Error en Gemini Stream: {e}"