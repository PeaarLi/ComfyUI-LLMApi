"""Node implementations for ComfyUI-LLMApi.

Every node class lives in its own module; this file is the single place that
lists them so the package entry point stays a two-liner. The order of
``ALL_NODES`` is the order the nodes appear in the Add Node menu.
"""

from __future__ import annotations

from .anthropic import LlmApiAnthropic
from .chat import LlmApiChat

ALL_NODES: list[type] = [
    LlmApiChat,
    LlmApiAnthropic,
]

__all__ = [
    "ALL_NODES",
    "LlmApiChat",
    "LlmApiAnthropic",
]
