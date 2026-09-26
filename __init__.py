"""ComfyUI-LLMApi -- talk to an LLM endpoint directly, with your own key.

Two nodes, one per protocol. In both, you supply the base URL, the API key and
the model name, and the node talks to that endpoint itself: no proxy, no Comfy
Org account, no credits.

* ``LlmApiChat`` -- the OpenAI chat-completions shape, which is the lingua franca
  of self-hosted servers and relays. Works against OpenAI, DeepSeek, Moonshot,
  Zhipu, DashScope, a local vLLM, Ollama's compatible layer, or any relay.
* ``LlmApiAnthropic`` -- the Anthropic Messages API, which the sibling cannot
  reach, and which is where Agent Skills live: either a local ``SKILL.md``
  injected into the system prompt, or skills mounted into Anthropic's own
  server-side container by id.

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
