"""Anthropic Messages API plumbing for ``LlmApiAnthropic``.

Same shape as ``_client.py``: pure functions for everything that can be pure, and
one thin async wrapper over the shared transport, so ``_selftest.py`` can drive
the whole request/response path against a local ``http.server`` stub.

The Messages API differs from chat-completions in ways that are easy to get
wrong, so each one is handled explicitly below rather than adapted from the
OpenAI shapes:

* Auth is a header *pair* (``x-api-key`` + ``anthropic-version``), and which one
  a custom endpoint wants is not guessable -- see ``_auth_headers``.
* ``max_tokens`` is required, not optional, so there is no "omit it" sentinel.
* Images are ``{"type": "image", "source": {"type": "base64", ...}}``, not
  ``data:`` URIs under ``image_url``.
* ``system`` is a top-level parameter, not a message with ``role: "system"``.
* The response is a ``content`` block array, not ``choices``, and the error
  envelope is nested under a ``type`` key rather than being the top-level key.

Raw HTTP rather than the ``anthropic`` SDK on purpose: the SDK is not installed
in this venv, and this pack's whole premise is that it needs nothing beyond what
ComfyUI already ships.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ._transport import _post_json, _snippet

__all__ = [
    "_AUTH_BEARER",
    "_AUTH_STYLES",
    "_AUTH_X_API_KEY",
    "_ERROR_PREFIX",
    "_MAX_NATIVE_SKILLS",
    "_auth_headers",
    "_base_headers",
    "_build_payload",
    "_build_system",
    "_build_user_content",
    "_messages_url",
    "_native_skill_entries",
    "_parse_response",
    "_post_messages",
]

_ERROR_PREFIX = "LlmApiAnthropic: "

_MESSAGES_SUFFIX = "/v1/messages"

# The two ways an Anthropic-shaped endpoint accepts a key. Official Anthropic
# wants x-api-key; a large share of relays only implement `Authorization: Bearer`
# because that is what every other OpenAI-compatible client sends. Sending both
# is not a safe middle ground -- gateways that reject unexpected headers exist --
# so the node picks one, and this is the combo's option list.
_AUTH_X_API_KEY = "x-api-key"
_AUTH_BEARER = "Authorization: Bearer"
_AUTH_STYLES = [_AUTH_X_API_KEY, _AUTH_BEARER]

# Skills that Anthropic ships itself, as opposed to ones the user published to
# POST /v1/skills. The two need different `type` values in the container, and the
# id is the only signal available: custom ids are minted as `skill_<...>`, and
# these four are documented by name. Guessing wrong produces a server error that
# names the skill, so a mistake is legible rather than silent.
_PRESET_SKILL_IDS = frozenset({"xlsx", "docx", "pptx", "pdf"})

# Anthropic caps a single request at 8 skills. Checked locally so the user gets a
# clear message instead of a 400 from the far end.
_MAX_NATIVE_SKILLS = 8

_CODE_EXECUTION_TOOL = "code_execution_20260521"
_BETA_CODE_EXECUTION = "code-execution-2025-08-25"
_BETA_SKILLS = "skills-2025-10-02"


def _messages_url(base_url: str) -> str:
    """Turn a user-supplied base URL into the messages endpoint.

    Handles both shapes people actually type: a bare host (which the API docs
    use, ``https://api.anthropic.com``) and a host already carrying ``/v1``
    (which is what the sibling OpenAI node trains you to type). Appending
    ``/v1/messages`` blindly would turn the second into ``/v1/v1/messages``.

    Like the OpenAI node, a missing version segment is never invented beyond
    that: relays put it in different places, and a wrong guess is a 404 that
    reads like a server fault. Pasting the complete ``.../messages`` URL works
    and is the escape hatch.
    """
    url = (base_url or "").strip()
    if not url:
        raise ValueError(
            f"{_ERROR_PREFIX}base_url 不能为空 / base_url is required. "
            "填端点前缀，例如 https://api.anthropic.com 或 https://api.anthropic.com/v1。"
        )
    url = url.rstrip("/")
    if url.endswith("/messages"):
        return url
    if url.endswith("/v1"):
        return url + "/messages"
    return url + _MESSAGES_SUFFIX


def _auth_headers(api_key: str, auth_style: str = _AUTH_X_API_KEY) -> dict[str, str]:
    """Build the one auth header this endpoint wants, or none at all.

    An empty key omits credentials entirely rather than sending an empty value,
    which is what lets a local Anthropic-compatible server run without a key --
    the same rule the OpenAI node follows.
    """
    key = (api_key or "").strip()
    if not key:
        return {}
    if auth_style == _AUTH_BEARER:
        return {"Authorization": f"Bearer {key}"}
    if auth_style == _AUTH_X_API_KEY:
        return {_AUTH_X_API_KEY: key}
    raise ValueError(
        f"{_ERROR_PREFIX}unknown auth_style {auth_style!r}; expected one of {_AUTH_STYLES}."
    )


def _base_headers(anthropic_version: str) -> dict[str, str]:
    """Content-Type plus the version header, which a few relays reject outright."""
    headers = {"Content-Type": "application/json"}
    version = (anthropic_version or "").strip()
    if version:
        headers["anthropic-version"] = version
    return headers


def _native_skill_entries(spec: str) -> list[dict[str, Any]]:
    """Parse the ``native_skills`` text into container entries.

    Accepts commas or newlines, so both ``xlsx, pptx`` and a column of ids work.
    """
    raw = (spec or "").replace("\n", ",")
    skill_ids = [item.strip() for item in raw.split(",") if item.strip()]
    if len(skill_ids) > _MAX_NATIVE_SKILLS:
        raise ValueError(
            f"{_ERROR_PREFIX}原生 skill 最多 {_MAX_NATIVE_SKILLS} 个 / at most {_MAX_NATIVE_SKILLS} "
            f"native skills per request; got {len(skill_ids)}. 这是 Anthropic 服务端的硬限制。"
        )
    return [
        {
            "type": "anthropic" if skill_id in _PRESET_SKILL_IDS else "custom",
            "skill_id": skill_id,
            "version": "latest",
        }
        for skill_id in skill_ids
    ]


def _build_system(skill_body: str, system_prompt: str) -> str:
    """Combine the selected skill's body with the user's own system prompt.

    The skill comes first and the user's instructions second, so that a user who
    wants to override something a skill says can simply say so afterwards --
    later text in the same system parameter wins.
    """
    parts: list[str] = []
    if (skill_body or "").strip():
        parts.append(skill_body.strip())
    if (system_prompt or "").strip():
        parts.append(system_prompt.strip())
    return "\n\n".join(parts)


def _build_user_content(prompt: str, image_payloads: list[tuple[str, str]]) -> Any:
    """The user turn: a plain string, or image blocks followed by the text block.

    Media before text, matching the sibling node and ComfyUI core's own OpenRouter
    node -- a vision model attends to an image better when the instruction that
    refers to it follows.
    """
    if not image_payloads:
        return prompt
    blocks: list[dict[str, Any]] = [
        {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": encoded}}
        for mime_type, encoded in image_payloads
    ]
    # Only add the text block when there is text. An image-only request is legal
    # ("describe this"), and an empty text block is the kind of thing strict
    # validators reject.
    if (prompt or "").strip():
        blocks.append({"type": "text", "text": prompt})
    return blocks


def _build_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    system: str,
    temperature: float,
    reasoning_trace: bool,
    native_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the request body.

    ``temperature`` is omitted below 0. That is not just tidiness: on the current
    models (Opus 5, Fable 5/5.1, Sonnet 5) sampling parameters were removed and
    sending one returns a 400, so the sentinel is the safe default and the tooltip
    says so.
    """
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if temperature >= 0:
        payload["temperature"] = float(temperature)
    if reasoning_trace:
        # `display` is the part that matters here. Thinking runs by default on the
        # current models, but its text comes back EMPTY unless summarized display
        # is asked for -- so without this the reasoning output would always be
        # blank, which looks like a bug rather than a default.
        payload["thinking"] = {"type": "adaptive", "display": "summarized"}
    if native_entries:
        payload["container"] = {"skills": native_entries}
        payload["tools"] = [{"type": _CODE_EXECUTION_TOOL, "name": "code_execution"}]
    return payload


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message")
        kind = error.get("type")
        if message and kind:
            return f"{message} ({kind})"
        if message:
            return str(message)
        return _snippet(json.dumps(error, ensure_ascii=False))
    return _snippet(str(error))


def _parse_response(body_text: str) -> tuple[str, str]:
    """Return ``(text, reasoning)`` from a Messages API response body."""
    if not (body_text or "").strip():
        raise ValueError(f"{_ERROR_PREFIX}endpoint 返回了空响应体 / endpoint returned an empty body.")

    try:
        data = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{_ERROR_PREFIX}响应不是合法 JSON / response was not valid JSON ({exc}). "
            f"这通常说明 base_url 指向的不是 Anthropic 兼容端点。原始响应 / raw body: {_snippet(body_text)}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"{_ERROR_PREFIX}响应顶层不是对象 / response was not a JSON object. 原始响应 / raw body: {_snippet(body_text)}"
        )

    error = data.get("error")
    if error:
        raise ValueError(f"{_ERROR_PREFIX}endpoint 返回错误 / endpoint returned an error: {_error_message(error)}")

    stop_reason = data.get("stop_reason")
    content = data.get("content")
    if not isinstance(content, list):
        raise ValueError(
            f"{_ERROR_PREFIX}响应里没有 content 数组 / response contained no content array "
            f"(顶层键 / top-level keys: {sorted(data)}). "
            "Anthropic 的响应形状是 content 块数组，不是 OpenAI 的 choices。"
            f" 原始响应 / raw body: {_snippet(body_text)}"
        )

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            value = block.get("text")
            if isinstance(value, str) and value:
                text_parts.append(value)
        elif kind == "thinking":
            value = block.get("thinking")
            if isinstance(value, str) and value.strip():
                thinking_parts.append(value)

    if stop_reason == "refusal":
        details = data.get("stop_details")
        category = details.get("category") if isinstance(details, dict) else None
        raise ValueError(
            f"{_ERROR_PREFIX}模型拒答 / model refused to respond"
            + (f" (category={category})" if category else "")
            + ". 这是安全分类器拒绝，不是配置问题。"
        )

    if stop_reason == "max_tokens":
        # Not fatal: there is still usable text, so it is returned with a warning
        # rather than thrown away.
        logging.warning(
            "%s回答因 max_tokens 被截断 / response truncated at max_tokens; raise max_tokens for a complete answer.",
            _ERROR_PREFIX,
        )

    if not text_parts and not thinking_parts:
        raise ValueError(
            f"{_ERROR_PREFIX}响应里没有可用的内容块 / response contained no usable content blocks "
            f"(stop_reason={stop_reason!r}). 原始响应 / raw body: {_snippet(body_text)}"
        )

    # Text blocks are contiguous fragments of one answer, so they are joined with
    # no separator; thinking blocks are distinct segments and get a blank line.
    return "".join(text_parts), "\n\n".join(thinking_parts)


# A misconfigured endpoint is the dominant failure mode, and the two auth styles
# are the most common way to get it wrong, so 401 leads with that.
_STATUS_HINTS = {
    400: (
        "请求被端点拒绝 / request rejected. 常见原因：model 名拼错；"
        "max_tokens 超出模型上限；temperature 在当前模型上已被移除（把它设为 -1 关掉）；"
        "skill_id 不存在或类型标错。"
    ),
    401: (
        "鉴权失败 / unauthorised. 官方 Anthropic 用 x-api-key 头；"
        "多数中转站只认 Authorization: Bearer —— 检查 auth_style 是否选对，"
        "以及 api_key 是否属于这个 base_url 对应的服务商。"
    ),
    403: "被拒绝访问 / forbidden. key 大多有效但无权访问该模型，或该地区被限制。",
    404: (
        "路径不存在 / not found. 官方端点是 /v1/messages；"
        "中转站的前缀各不相同，可以直接把完整的 .../messages 粘进 base_url。"
    ),
    413: "请求体过大 / payload too large. 减少图片数量，或先把图缩小再接进来。",
    429: "触发限流 / rate limited. 稍后重试，或调大 max_retries。",
}

# Appended when a container-path request fails, because the most likely cause is
# not the user's key or model but the endpoint: the server-side container is
# Anthropic's own infrastructure and a relay has to pass it through to work.
_NATIVE_SKILL_NOTE = (
    "你这次用了原生 skill（容器路径）—— 该路径需要端点透传 Anthropic 的服务端容器，"
    "多数中转站做不到。自定义 BaseURL 时请改用本地 skill（skill 参数）。"
)


def _status_hint(status: int, native_active: bool = False) -> str:
    hint = _STATUS_HINTS.get(status, "")
    if native_active and status in (400, 404):
        return f"{_NATIVE_SKILL_NOTE} {hint}".strip()
    if not hint and 300 <= status < 400:
        return "端点重定向后没有落到最终地址 / redirected without reaching a final URL."
    if not hint and status >= 500:
        return "端点侧故障 / server-side failure. 可以调大 max_retries 稍后重试。"
    return hint


async def _post_messages(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float,
    max_retries: int,
    native_active: bool = False,
) -> str:
    """POST to a messages endpoint via the shared transport."""
    return await _post_json(
        url,
        headers,
        payload,
        timeout=timeout,
        max_retries=max_retries,
        error_prefix=_ERROR_PREFIX,
        status_hint=lambda status: _status_hint(status, native_active),
    )
