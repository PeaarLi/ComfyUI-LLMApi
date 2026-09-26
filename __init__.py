"""ComfyUI-LLMApi -- talk to any OpenAI-compatible chat endpoint with your own key.

One node, ``LlmApiChat``: you supply the base URL, the API key and the model
name, and this pack talks to that endpoint directly. No proxy, no Comfy Org
account, no credits -- so it works against OpenAI, DeepSeek, Moonshot, Zhipu,
DashScope, a local vLLM, Ollama's OpenAI-compatible layer, or any relay, as long
as the endpoint speaks the OpenAI chat-completions shape.

This package uses the **V3 node schema** (``comfy_api.latest``), not the legacy
V1 ``NODE_CLASS_MAPPINGS`` dict. ComfyUI picks the V3 path only when
``NODE_CLASS_MAPPINGS`` is *absent* from this module: ``load_custom_node``
(nodes.py) checks ``hasattr(module, "NODE_CLASS_MAPPINGS")`` first, so defining
it -- even as an empty dict -- routes the pack down the V1 branch and silently
skips ``comfy_entrypoint``. The pack then vanishes with no error at all. Do not
add it.

Everything importable lives in ``nodes/``; this file is deliberately just the
entry point.
"""

from __future__ import annotations

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import ALL_NODES

__all__ = ["LlmApiExtension", "comfy_entrypoint"]


class LlmApiExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(ALL_NODES)


async def comfy_entrypoint() -> LlmApiExtension:
    """ComfyUI calls this to load the extension and its nodes."""
    return LlmApiExtension()
