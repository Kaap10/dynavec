"""Chat model abstraction and provider implementations."""

from __future__ import annotations

from typing import Any

from dynavec.chat.base import (
    ChatChunk,
    ChatModel,
    ChatResult,
    Message,
    Role,
    Tool,
    ToolCall,
)

__all__ = [
    "ChatChunk",
    "ChatModel",
    "ChatResult",
    "Message",
    "Role",
    "Tool",
    "ToolCall",
]


# Optional imports for providers
def _import_openai() -> Any:
    from dynavec.chat.openai import OpenAIChatModel

    return OpenAIChatModel


def _import_anthropic() -> Any:
    from dynavec.chat.anthropic import AnthropicChatModel

    return AnthropicChatModel


def _import_bedrock() -> Any:
    from dynavec.chat.bedrock import BedrockChatModel

    return BedrockChatModel


def __getattr__(name: str) -> Any:
    if name == "OpenAIChatModel":
        return _import_openai()
    if name == "AnthropicChatModel":
        return _import_anthropic()
    if name == "BedrockChatModel":
        return _import_bedrock()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__.extend(["OpenAIChatModel", "AnthropicChatModel", "BedrockChatModel"])
