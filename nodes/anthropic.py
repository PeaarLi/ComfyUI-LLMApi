"""``LlmApiAnthropic`` -- the Anthropic Messages API, with skills.

The sibling ``LlmApiChat`` node speaks the OpenAI chat-completions shape, which
is the lingua franca of self-hosted servers and relays but reaches none of
Anthropic's own features. This node speaks the Messages API instead, and exists
mainly for one of those features: **Agent Skills**.

Skills reach the model by two different routes, and the node keeps them visibly
separate because they are not interchangeable:

* ``skill`` -- a local ``SKILL.md`` from this pack's ``skills/`` directory, whose
  body is injected into the ``system`` parameter, followed by its ``references/``
  files. Works against **any** Anthropic-shaped endpoint, including relays,
  because it is only text. This is what you want when you point ``base_url`` at a
  relay. Inlining the references matters more than it sounds: many skills are
  routers whose body says "read references/x.txt and follow it", so the body alone
  would instruct the model to read a file it has no way to reach.
* ``native_skills`` -- skill ids attached as ``container.skills`` alongside the
  code-execution tool, which is Anthropic's own mechanism: the skills are mounted
  into a server-side sandbox and the model reads them from there. More capable,
  but it needs an endpoint that passes Anthropic's container infrastructure
  through, which a relay generally does not.

Both may be set at once; they are independent.

Skills are not sent as a system prompt by either route in Anthropic's own
implementation -- the container route has the model read ``SKILL.md`` on demand.
The local route is an approximation of that, and is documented as one.
"""

from __future__ import annotations

from comfy_api.latest import io

from ._anthropic_client import (
    _AUTH_STYLES,
    _AUTH_X_API_KEY,
    _BETA_CODE_EXECUTION,
    _BETA_SKILLS,
    _auth_headers,
    _base_headers,
    _build_payload,
    _build_system,
    _build_user_content,
    _messages_url,
    _native_skill_entries,
    _parse_response,
    _post_messages,
)
from ._media import _image_payloads, _select_image_tensors
from ._skills import _NO_SKILL, _discover_skills, _load_skill_body

__all__ = ["LlmApiAnthropic"]

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_REASONING_TRACE = True
DEFAULT_TIMEOUT = 300.0
DEFAULT_MAX_RETRIES = 0

# Anthropic requires max_tokens, so this is a real value rather than an "omit it"
# sentinel like the sibling node's 0. 16000 is the documented starting point for
# a non-streaming request: large enough that a normal answer never truncates,
# small enough to stay inside the SDK/HTTP timeout the docs assume.
DEFAULT_MAX_TOKENS = 16000
MAX_TOKENS_MIN = 1
MAX_TOKENS_MAX = 128000

# -1 means "do not send the field". Sampling parameters were REMOVED from the
# current models (Opus 5, Fable 5/5.1, Sonnet 5) and sending one returns a 400,
# so omitting is the safe default. Note the range is 0..1, not the 0..2 the
# OpenAI node uses -- Anthropic never accepted 2.
DEFAULT_TEMPERATURE = -1.0
TEMPERATURE_MIN = -1.0
TEMPERATURE_MAX = 1.0

TIMEOUT_MIN = 1.0
TIMEOUT_MAX = 3600.0
RETRIES_MAX = 10

MAX_IMAGES = 8
IMAGE_SLOT_NAMES = [f"image_{index}" for index in range(1, MAX_IMAGES + 1)]

DEFAULT_INLINE_REFERENCES = True

# Discovered once, at import: the combo's options cannot change while ComfyUI is
# running, so a skill directory added later needs a restart. Bodies are re-read on
# every call, so editing one does not. _selftest.py pins both halves of that.
SKILLS = _discover_skills()
SKILL_OPTIONS = [_NO_SKILL] + [skill.name for skill in SKILLS]

# Enough of each description to choose by, without the tooltip becoming a wall of
# text. The bundled set is nine skills whose descriptions run to several hundred
# characters each; untruncated, that is a ~3 KB hover.
_LISTING_DESCRIPTION_CHARS = 90


def _skill_listing() -> str:
    """One line naming every bundled skill, for the combo's tooltip."""
    if not SKILLS:
        return "（包内没有 skill / no skills bundled）"
    parts = []
    for skill in SKILLS:
        description = " ".join(skill.description.split())
        if len(description) > _LISTING_DESCRIPTION_CHARS:
            description = description[: _LISTING_DESCRIPTION_CHARS - 1].rstrip() + "…"
        parts.append(f"{skill.name} — {description}")
    return "; ".join(parts)


class LlmApiAnthropic(io.ComfyNode):
    """Send a system+user request to any Anthropic-shaped endpoint, with skills."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        skill_listing = _skill_listing()
        return io.Schema(
            node_id="LlmApiAnthropic",
            display_name="LLM API Anthropic",
            category="text/llm api",
            search_aliases=["Anthropic", "Claude", "skill", "SKILL.md", "LLM", "chat", "api", "base url"],
            description=(
                "Send a request to an Anthropic Messages API endpoint using your own base URL and key, with "
                "optional skills. Skills come in two independent forms: a local SKILL.md injected into the "
                "system prompt (works on any endpoint, including relays), or native Anthropic skills mounted "
                "into a server-side container (needs an endpoint that passes the container through)."
            ),
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    dynamic_prompts=True,
                    tooltip=(
                        "The user message. May be empty when at least one image is wired, but prompt and "
                        "images cannot both be empty."
                    ),
                ),
                io.String.Input(
                    "model",
                    default=DEFAULT_MODEL,
                    placeholder="claude-opus-5",
                    tooltip=(
                        "Model name exactly as the endpoint expects it: claude-opus-5, claude-sonnet-5, "
                        "claude-haiku-4-5, or whatever a relay calls them. Free text because a relay's model "
                        "names cannot be known in advance."
                    ),
                ),
                io.String.Input(
                    "base_url",
                    optional=True,
                    default=DEFAULT_BASE_URL,
                    placeholder="https://api.anthropic.com/v1",
                    tooltip=(
                        "Endpoint prefix. Resolved like this: a bare host gets '/v1/messages' appended; a host "
                        "ending in '/v1' gets '/messages'; anything already ending in '/messages' is used as-is. "
                        "So both https://api.anthropic.com and https://api.anthropic.com/v1 work, and pasting "
                        "the complete URL is the escape hatch for a relay whose paths differ."
                    ),
                ),
                io.String.Input(
                    "api_key",
                    optional=True,
                    default="",
                    tooltip=(
                        "Key for the endpoint. Empty sends no auth header at all, which is what a local "
                        "Anthropic-compatible server needs. WARNING: ComfyUI has no password field, so this is "
                        "stored as plain text in the workflow JSON and visible on screen."
                    ),
                ),
                io.Combo.Input(
                    "auth_style",
                    optional=True,
                    options=_AUTH_STYLES,
                    default=_AUTH_X_API_KEY,
                    advanced=True,
                    tooltip=(
                        "Which header carries the key. Official Anthropic wants 'x-api-key'; most relays only "
                        "implement 'Authorization: Bearer' because that is what every other OpenAI-compatible "
                        "client sends. Only one is ever sent -- a 401 is the usual symptom of picking wrong."
                    ),
                ),
                io.String.Input(
                    "anthropic_version",
                    optional=True,
                    default=DEFAULT_ANTHROPIC_VERSION,
                    advanced=True,
                    tooltip=(
                        "Value of the 'anthropic-version' header. Blank omits the header entirely, which a few "
                        "relays prefer."
                    ),
                ),
                io.String.Input(
                    "system_prompt",
                    optional=True,
                    multiline=True,
                    default="",
                    tooltip=(
                        "Your instructions, sent as the top-level 'system' parameter. Sent after the selected "
                        "skill's text, so anything you write here overrides the skill."
                    ),
                ),
                io.Combo.Input(
                    "skill",
                    optional=True,
                    options=SKILL_OPTIONS,
                    default=_NO_SKILL,
                    tooltip=(
                        "A SKILL.md from this pack's skills/ directory, injected into the system parameter. "
                        f"Bundled -- {skill_listing}. "
                        "Works on any endpoint including relays, because it is only text. Its references/ files "
                        "are inlined too, which is what makes a router-style skill work -- but the model still "
                        "cannot execute a skill's scripts/. The list is scanned at startup, so a skill directory "
                        "added later needs a ComfyUI restart; editing an existing skill's text does not."
                    ),
                ),
                io.Boolean.Input(
                    "inline_references",
                    optional=True,
                    default=DEFAULT_INLINE_REFERENCES,
                    advanced=True,
                    tooltip=(
                        "Append the selected skill's references/ files to the system prompt. On by default "
                        "because many skills are routers -- their SKILL.md tells the reader to go read a file "
                        "under references/, which a single-request node cannot do, so without this the model is "
                        "told to read something it cannot reach. Turn it off to save context when you know the "
                        "body stands alone, or when the references are large."
                    ),
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
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
                        f"Optional reference images for vision models, up to {MAX_IMAGES}. Each is downscaled to "
                        "fit 1280x1280 and sent as a base64 image block -- Anthropic's own shape, not OpenAI's "
                        "data URI. A batched input counts one image per frame."
                    ),
                ),
                io.Boolean.Input(
                    "reasoning_trace",
                    optional=True,
                    default=DEFAULT_REASONING_TRACE,
                    advanced=True,
                    tooltip=(
                        "Ask for the model's thinking to be returned in the 'reasoning' output. This sets "
                        "thinking.display to 'summarized'. Turn it off and the field is not sent -- note that "
                        "thinking still runs on the current models, its text just is not returned, so you would "
                        "only be saving the tokens the summary would have cost."
                    ),
                ),
                io.String.Input(
                    "native_skills",
                    optional=True,
                    default="",
                    multiline=True,
                    advanced=True,
                    placeholder="xlsx, skill_abc123",
                    tooltip=(
                        "Anthropic-native skills, by id, comma or newline separated (max 8 per request). These "
                        "are mounted into a server-side container with the code-execution tool, so the model can "
                        "read their supporting files -- the real mechanism, unlike the 'skill' dropdown above. "
                        "Ids are either Anthropic's prebuilt document skills (xlsx, docx, pptx, pdf) or ones you "
                        "published to POST /v1/skills; other ids are sent as custom. WARNING: the container is "
                        "Anthropic's own infrastructure, so a relay will usually fail here -- on a custom "
                        "base_url use the 'skill' dropdown instead."
                    ),
                ),
                io.Int.Input(
                    "max_tokens",
                    optional=True,
                    default=DEFAULT_MAX_TOKENS,
                    min=MAX_TOKENS_MIN,
                    max=MAX_TOKENS_MAX,
                    step=1,
                    tooltip=(
                        "Required by Anthropic -- there is no 'leave it to the server' option here. Budget for "
                        "the thinking trace too: a reasoning model spends part of this before writing any answer, "
                        "and hitting the cap truncates the reply (the node warns rather than failing)."
                    ),
                ),
                io.Float.Input(
                    "temperature",
                    optional=True,
                    default=DEFAULT_TEMPERATURE,
                    min=TEMPERATURE_MIN,
                    max=TEMPERATURE_MAX,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Sampling temperature, range 0-1 (not the 0-2 the OpenAI node accepts). The default -1 "
                        "means the field is not sent at all. That default is deliberate: sampling parameters were "
                        "removed from the current models and sending one returns a 400."
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
                        "Seconds to wait for the response, covering the model's whole generation -- including "
                        "the thinking trace and, on the native-skill path, whatever runs inside the container."
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
                        "422) are never retried."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip="The model's reply, from the text blocks of the response. Empty if it returned only a trace.",
                ),
                io.String.Output(
                    display_name="reasoning",
                    tooltip=(
                        "The thinking blocks, when the endpoint returns them. Empty when reasoning_trace is off, "
                        "or for a model that does not return a trace."
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
        auth_style: str = _AUTH_X_API_KEY,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        system_prompt: str = "",
        skill: str = _NO_SKILL,
        inline_references: bool = DEFAULT_INLINE_REFERENCES,
        images: dict | None = None,
        reasoning_trace: bool = DEFAULT_REASONING_TRACE,
        native_skills: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> io.NodeOutput:
        model = (model or "").strip()
        if not model:
            raise ValueError(
                "LlmApiAnthropic: model 不能为空 / model is required. "
                "常见值：claude-opus-5、claude-sonnet-5、claude-haiku-4-5（中转站可能用自己的名字）。"
            )

        max_tokens = int(max_tokens)
        if max_tokens < MAX_TOKENS_MIN:
            raise ValueError(
                f"LlmApiAnthropic: max_tokens 必须 >= {MAX_TOKENS_MIN} / must be at least {MAX_TOKENS_MIN}; "
                f"got {max_tokens}. Anthropic 要求这个字段必填。"
            )

        image_tensors = _select_image_tensors(images, IMAGE_SLOT_NAMES)
        if len(image_tensors) > MAX_IMAGES:
            raise ValueError(
                f"LlmApiAnthropic: 最多 {MAX_IMAGES} 张图片 / at most {MAX_IMAGES} images are supported; "
                f"got {len(image_tensors)}. 注意批量输入会按帧展开 / note a batched input expands one image per frame."
            )
        if not (prompt or "").strip() and not image_tensors:
            raise ValueError(
                "LlmApiAnthropic: prompt 和 images 不能同时为空 / prompt and images cannot both be empty."
            )

        skill_body = _load_skill_body(skill, inline_references=bool(inline_references))
        native_entries = _native_skill_entries(native_skills)

        headers = _base_headers(anthropic_version) | _auth_headers(api_key, auth_style)
        if native_entries:
            # Both betas are required together for the container route: one enables
            # the code-execution tool the skills run in, the other the skills API.
            headers["anthropic-beta"] = ",".join([_BETA_CODE_EXECUTION, _BETA_SKILLS])

        payload = _build_payload(
            model,
            [{"role": "user", "content": _build_user_content(prompt, _image_payloads(image_tensors))}],
            max_tokens=max_tokens,
            system=_build_system(skill_body, system_prompt),
            temperature=float(temperature),
            reasoning_trace=bool(reasoning_trace),
            native_entries=native_entries,
        )
        body = await _post_messages(
            _messages_url(base_url),
            headers,
            payload,
            timeout=float(timeout),
            max_retries=int(max_retries),
            native_active=bool(native_entries),
        )
        text, reasoning = _parse_response(body)
        return io.NodeOutput(text, reasoning)
