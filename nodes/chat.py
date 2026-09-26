"""``LlmApiChat`` -- call any OpenAI-compatible chat endpoint with your own key.

ComfyUI's built-in text nodes all route through Comfy Org: they talk to
``api.comfy.org``, take the account token from the frontend, and bill credits.
That is fine for the curated models, but it cannot reach a self-hosted vLLM, an
Ollama box on the LAN, or a relay whose model list nobody has enumerated. This
node is the other half of that: you type the base URL, the key and the model
name, and it posts to ``<base_url>/chat/completions`` directly.

The node is intentionally a thin orchestrator -- URL joining, request shaping,
parsing and media encoding live in ``_client.py`` and ``_media.py``, which is
what makes them testable without a server or a network.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._client import (
    _auth_headers,
    _build_messages,
    _build_payload,
    _chat_completions_url,
    _parse_response,
    _post_chat,
)
from ._media import _images_to_data_uris, _select_image_tensors

__all__ = ["LlmApiChat"]

# Schema defaults, kept in one place so ``define_schema`` and the execute-side
# unpacking cannot drift apart. ``_selftest.py`` asserts the schema defaults
# equal these.
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = ""
DEFAULT_SEED = 0
DEFAULT_MAX_RETRIES = 0
DEFAULT_TIMEOUT = 300.0

# Sentinel defaults meaning "do not send this field at all", so the endpoint
# applies its own default. 0 is a legal temperature and 0 max_tokens is not a
# meaningful request, which is why the sentinel is -1 for one and 0 for the
# other rather than a shared value. The same 0-means-omit convention ComfyUI
# core uses for ``seed`` in its OpenRouter node.
DEFAULT_TEMPERATURE = -1.0
DEFAULT_MAX_TOKENS = 0

SEED_MAX = 2147483647
MAX_TOKENS_MAX = 1000000
TIMEOUT_MIN = 1.0
TIMEOUT_MAX = 3600.0
RETRIES_MAX = 10

# Autogrow image slots. Each one is a separate reference image the model sees in
# declared order, and each is inlined into the request as base64, so this is
# capped well below the engine's own limit of 100 names.
MAX_IMAGES = 8
IMAGE_SLOT_NAMES = [f"image_{index}" for index in range(1, MAX_IMAGES + 1)]


class LlmApiChat(io.ComfyNode):
    """Send a system+user chat request to any OpenAI-compatible endpoint."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LlmApiChat",
            display_name="LLM API Chat",
            category="text/llm api",
            search_aliases=["LLM", "OpenAI", "GPT", "chat", "api", "base url", "DeepSeek", "Ollama"],
            description=(
                "Send a chat request to any OpenAI-compatible endpoint using your own base URL and API key: "
                "OpenAI, DeepSeek, Moonshot, Zhipu, DashScope, a local vLLM/Ollama, or a relay. Supports "
                "optional reference images for vision models, optional system prompt, and returns the model's "
                "reasoning trace separately when it produces one."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    dynamic_prompts=True,
                    tooltip=(
                        "The user message. May be empty when at least one image is wired (e.g. 'describe this'), "
                        "but prompt and images cannot both be empty."
                    ),
                ),
                io.String.Input(
                    "model",
                    default=DEFAULT_MODEL,
                    placeholder="gpt-4o-mini",
                    tooltip=(
                        "Model name exactly as the endpoint expects it -- e.g. gpt-4o-mini, deepseek-chat, "
                        "moonshot-v1-8k, qwen-vl-max, llama3.2-vision. A free-text field on purpose: this node "
                        "can reach endpoints whose model list cannot be known in advance."
                    ),
                ),
                io.String.Input(
                    "base_url",
                    optional=True,
                    default=DEFAULT_BASE_URL,
                    placeholder="https://api.deepseek.com/v1",
                    tooltip=(
                        "Endpoint prefix. '/chat/completions' is appended unless it is already there, so give "
                        "the part up to and including the version: https://api.openai.com/v1, "
                        "https://api.deepseek.com/v1, http://127.0.0.1:11434/v1. A missing '/v1' is NOT guessed "
                        "-- relays differ, and a wrong guess is a confusing 404."
                    ),
                ),
                io.String.Input(
                    "api_key",
                    optional=True,
                    default="",
                    tooltip=(
                        "Bearer token for the endpoint. Leave empty for local servers that need no key (Ollama, "
                        "vLLM, LM Studio) -- no Authorization header is sent at all in that case. WARNING: "
                        "ComfyUI has no password field, so this value is stored as plain text in the workflow "
                        "JSON and is visible on screen. Do not share or commit workflows containing a key."
                    ),
                ),
                io.String.Input(
                    "system_prompt",
                    optional=True,
                    multiline=True,
                    default="",
                    tooltip="Foundational instructions that shape the model's behaviour. Blank omits the system message entirely.",
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        # This inner tooltip is load-bearing, and it is the one
                        # string in the schema that locales/zh CANNOT override:
                        # the frontend names a child slot "images.image_1" and
                        # looks it up as "images_image_1", finds nothing in
                        # nodeDefs.json, and falls back to whatever the backend
                        # declared here. Leave it empty and every child slot in
                        # every language shows a blank tooltip. Kept bilingual
                        # for that reason. _selftest.py asserts it is non-empty.
                        io.Image.Input(
                            "image",
                            tooltip=(
                                "Image frame, [H, W, C]. Batched input expands one image per frame. "
                                "单帧图像；批量输入会按帧展开成多张。"
                            ),
                        ),
                        names=IMAGE_SLOT_NAMES,
                        min=0,
                    ),
                    tooltip=(
                        f"Optional reference images for vision models, up to {MAX_IMAGES}. They reach the model "
                        "in slot order, so 'compare image_1 with image_2' means what it says. Each is downscaled "
                        "to fit 1280x1280 and inlined as a PNG data URI; a batched input counts one image per "
                        "frame. The model must actually accept images -- a text-only model will reject the request."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    optional=True,
                    default=DEFAULT_SEED,
                    min=0,
                    max=SEED_MAX,
                    step=1,
                    control_after_generate=True,
                    tooltip=(
                        "Sent to the endpoint only when above 0, so the default omits it. Its main job is "
                        "ComfyUI-side: with 'control_after_generate' set to randomize, a changing seed is what "
                        "makes this node re-run instead of replaying its cached output."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    optional=True,
                    default=DEFAULT_TEMPERATURE,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Sampling temperature. The default -1 means the field is not sent at all and the "
                        "endpoint's own default applies -- 0 is a real temperature (greedy), so it cannot be "
                        "the sentinel. Some reasoning models reject this field outright."
                    ),
                ),
                io.Int.Input(
                    "max_tokens",
                    optional=True,
                    default=DEFAULT_MAX_TOKENS,
                    min=0,
                    max=MAX_TOKENS_MAX,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Upper bound on generated tokens. 0 (the default) omits the field and lets the endpoint "
                        "decide. Raise it for long outputs -- reasoning models spend a lot of the budget on the "
                        "trace before writing any answer."
                    ),
                ),
                io.Float.Input(
                    "timeout",
                    optional=True,
                    default=DEFAULT_TIMEOUT,
                    min=TIMEOUT_MIN,
                    max=TIMEOUT_MAX,
                    step=1.0,
                    advanced=True,
                    tooltip=(
                        "Seconds to wait for the response. This is a ceiling on the whole request, so it must "
                        "cover the model's full generation time -- 300 is generous for chat, but a long "
                        "reasoning trace or a large image batch may need more."
                    ),
                ),
                io.Int.Input(
                    "max_retries",
                    optional=True,
                    default=DEFAULT_MAX_RETRIES,
                    min=0,
                    max=RETRIES_MAX,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Retries after a connection failure, timeout, or a transient status (429, 5xx), with "
                        "exponential backoff. 0 sends the request exactly once. Configuration errors (401, 404, "
                        "422) are never retried -- they would fail identically every time."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip="The model's reply. Empty string if it produced only a reasoning trace.",
                ),
                io.String.Output(
                    display_name="reasoning",
                    tooltip=(
                        "The model's thinking, when the endpoint returns one. Empty string for models that do "
                        "not think out loud, so this port is always safe to wire up."
                    ),
                ),
            ],
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        system_prompt: str = "",
        images: dict | None = None,
        seed: int = DEFAULT_SEED,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> io.NodeOutput:
        model = (model or "").strip()
        if not model:
            raise ValueError(
                "LlmApiChat: model 不能为空 / model is required. "
                "填入端点认识的模型名，例如 gpt-4o-mini、deepseek-chat、qwen-vl-max。"
            )

        image_tensors = _select_image_tensors(images, IMAGE_SLOT_NAMES)
        if len(image_tensors) > MAX_IMAGES:
            raise ValueError(
                f"LlmApiChat: 最多 {MAX_IMAGES} 张图片 / at most {MAX_IMAGES} images are supported; "
                f"got {len(image_tensors)}. 注意批量输入会按帧展开 / note a batched input expands one image per frame."
            )
        if not (prompt or "").strip() and not image_tensors:
            raise ValueError(
                "LlmApiChat: prompt 和 images 不能同时为空 / prompt and images cannot both be empty. "
                "至少给一个 prompt，或者接一张图片。"
            )

        url = _chat_completions_url(base_url)
        messages = _build_messages(system_prompt, prompt, _images_to_data_uris(image_tensors))
        payload = _build_payload(
            model,
            messages,
            seed=int(seed),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        body = await _post_chat(
            url,
            _auth_headers(api_key),
            payload,
            timeout=float(timeout),
            max_retries=int(max_retries),
        )
        text, reasoning = _parse_response(body)
        return io.NodeOutput(text, reasoning)
