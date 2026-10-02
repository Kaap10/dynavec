"""Chat model abstraction and provider implementations."""

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
def _import_openai():
    from dynavec.chat.openai import OpenAIChatModel
    return OpenAIChatModel

def _import_anthropic():
    from dynavec.chat.anthropic import AnthropicChatModel
    return AnthropicChatModel

def _import_bedrock():
    from dynavec.chat.bedrock import BedrockChatModel
    return BedrockChatModel

def __getattr__(name: str):
    if name == "OpenAIChatModel":
        return _import_openai()
    elif name == "AnthropicChatModel":
        return _import_anthropic()
    elif name == "BedrockChatModel":
        return _import_bedrock()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__.extend(["OpenAIChatModel", "AnthropicChatModel", "BedrockChatModel"])
