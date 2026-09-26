"""HTTP plumbing for ``LlmApiChat``: URL joining, request building, response parsing.

Everything here is either a pure function or a thin async wrapper around
``httpx``, which is what lets ``_selftest.py`` drive the entire
request/response path against a local ``http.server`` stub -- no ComfyUI server,
no network, no API key needed to prove the node behaves.

Every user-facing failure message is built here so they all carry the same
``LlmApiChat: `` prefix and the same 中英混排 register as the rest of the pack.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from ._media import _reasoning_from_message

__all__ = [
    "_ERROR_PREFIX",
    "_auth_headers",
    "_build_messages",
    "_build_payload",
    "_chat_completions_url",
    "_parse_response",
    "_post_chat",
]

_ERROR_PREFIX = "LlmApiChat: "

_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"

# Enough of a bad response body to diagnose the problem without dumping a
# megabyte of HTML error page into ComfyUI's log and error box.
_SNIPPET_CHARS = 600

# Statuses worth a retry: the transient ones. Everything else (401, 404, 422...)
# is a configuration error that will fail identically however many times it is
# sent, so retrying only delays the error the user needs to see.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRY_BASE_DELAY = 1.0

# Redirects that instruct clients to re-issue the request as a GET, dropping the
# body (RFC 9110). 307/308 are deliberately absent: they preserve method and body.
_METHOD_REWRITING_STATUS = frozenset({301, 302, 303})

# This node's dominant failure mode is a misconfigured endpoint, not a broken
# server -- the user supplies the URL, the key and the model by hand, and every
# one of them can be subtly wrong. So the common statuses carry their fix
# inline rather than making the reader go and look it up.
_STATUS_HINTS = {
    400: (
        "请求被端点拒绝 / request rejected. 常见原因：model 名拼错；端点不认某个参数"
        "（temperature 设为 -1、max_tokens 设为 0 可以关掉这两个）；图片过大。"
    ),
    401: (
        "鉴权失败 / unauthorised. 检查 api_key 是否属于这个 base_url 对应的服务商 —— "
        "把 DeepSeek 的 key 发到 api.openai.com 就是这个结果。"
    ),
    403: (
        "被拒绝访问 / forbidden. key 大多有效但无权访问该模型，或该地区被限制。"
    ),
    404: (
        "路径不存在 / not found. base_url 一般要带版本段（/v1、/v4、/compatible-mode/v1），"
        "也可以直接把完整的 .../chat/completions 粘进来。"
    ),
    413: "请求体过大 / payload too large. 减少图片数量，或先把图缩小再接进来。",
    422: "参数校验失败 / unprocessable. 端点不接受某个参数或取值，检查 model 名与图片格式。",
    429: "触发限流 / rate limited. 稍后重试，或调大 max_retries。",
}


def _redirect_note(response: httpx.Response) -> str:
    """Name a followed redirect that rewrote POST into GET.

    Redirects ARE followed, because 307/308 preserve the method and body and a
    relay using one is genuinely rescued by that. But 301/302/303 tell every
    compliant client to re-issue the request as a GET and drop the body, so a
    chat-completions call arriving that way cannot succeed. What comes back --
    405, 501, or a login page -- describes the destination, not the problem,
    and a status hint would blame the wrong party. The chain is still on
    ``response.history``, so the real cause can be stated instead.
    """
    rewritten = [item for item in response.history if item.status_code in _METHOD_REWRITING_STATUS]
    if not rewritten:
        return ""
    first = rewritten[0]
    location = first.headers.get("location", "未知地址 / unknown")
    return (
        f"请求经过了 {first.status_code} 重定向到 {location}，而 301/302/303 会把 POST 改写成 GET 并丢掉请求体，"
        "所以端点不可能正常返回。请把 base_url 直接改成重定向后的最终地址。"
    )


def _status_hint(status: int) -> str:
    hint = _STATUS_HINTS.get(status)
    if hint:
        return hint
    if 300 <= status < 400:
        # follow_redirects is on, so a 3xx surviving to here means the chain
        # ended without reaching a real endpoint -- typically a relay bouncing
        # to a login page.
        return "端点重定向后没有落到最终地址 / redirected without reaching a final URL. 常见于中转站跳到登录页，请改用最终地址。"
    if status >= 500:
        return "端点侧故障 / server-side failure. 可以调大 max_retries 稍后重试。"
    return ""


def _chat_completions_url(base_url: str) -> str:
    """Turn a user-supplied base URL into the chat-completions endpoint.

    Deliberately does NOT guess a missing ``/v1``. Relays differ on whether the
    prefix is ``/v1``, ``/openai/v1``, ``/compatible-mode/v1`` or absent at all,
    so a wrong guess produces a 404 that looks like a server problem rather than
    a typo. The rule is mechanical and documented in the tooltip and README:
    append ``/chat/completions`` unless it is already there.
    """
    url = (base_url or "").strip()
    if not url:
        raise ValueError(
            f"{_ERROR_PREFIX}base_url 不能为空 / base_url is required. "
            "填端点前缀，例如 https://api.openai.com/v1 —— 本节点会在后面接上 /chat/completions。"
        )
    url = url.rstrip("/")
    if url.endswith(_CHAT_COMPLETIONS_SUFFIX):
        return url
    return url + _CHAT_COMPLETIONS_SUFFIX


def _auth_headers(api_key: str) -> dict[str, str]:
    """Bearer header, or no header at all.

    An empty key omits the header entirely rather than sending
    ``Authorization: Bearer `` -- local servers (Ollama, vLLM, LM Studio) need no
    key, and an empty-but-present header is a 401 on some of them.
    """
    key = (api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _build_messages(system_prompt: str, prompt: str, image_data_uris: list[str]) -> list[dict[str, Any]]:
    """Assemble the ``messages`` array from the node's fixed fields.

    Media blocks come before the text block, matching ComfyUI core's own
    OpenRouter node (``_user_message`` in comfy_api_nodes/nodes_openrouter.py).
    Vision models generally attend to an image better when the instruction that
    refers to it follows rather than precedes it.
    """
    messages: list[dict[str, Any]] = []
    if (system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt})
    if image_data_uris:
        content: Any = [{"type": "image_url", "image_url": {"url": uri}} for uri in image_data_uris]
        content.append({"type": "text", "text": prompt})
    else:
        # A plain string, not a one-element block list: some OpenAI-compatible
        # servers still reject the array form for text-only requests.
        content = prompt
    messages.append({"role": "user", "content": content})
    return messages


def _build_payload(
    model: str,
    messages: list[dict[str, Any]],
    *,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Build the request body, omitting every optional field left at its sentinel.

    ``temperature = -1`` and ``max_tokens = 0`` are the "leave it to the server"
    sentinels (see the constants in ``chat.py``). Omitting them matters: a
    temperature sent as 0 is not the same request as no temperature, and several
    reasoning models reject ``temperature`` outright.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        # Streaming is never used: the node returns whole strings, and ComfyUI's
        # node boundary has nowhere to put a partial one.
        "stream": False,
    }
    if seed > 0:
        payload["seed"] = int(seed)
    if temperature >= 0:
        payload["temperature"] = float(temperature)
    if max_tokens > 0:
        payload["max_tokens"] = int(max_tokens)
    return payload


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "…"


def _error_message(error: Any) -> str:
    """Pull a readable message out of an ``error`` field of unknown shape."""
    if isinstance(error, dict):
        message = error.get("message") or error.get("detail") or error.get("type")
        code = error.get("code")
        if message and code is not None:
            return f"{message} (code {code})"
        if message:
            return str(message)
        return _snippet(json.dumps(error, ensure_ascii=False))
    return _snippet(str(error))


def _parse_response(body_text: str) -> tuple[str, str]:
    """Return ``(text, reasoning)`` from a chat-completions response body."""
    if not (body_text or "").strip():
        raise ValueError(f"{_ERROR_PREFIX}endpoint 返回了空响应体 / endpoint returned an empty body.")

    try:
        data = json.loads(body_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{_ERROR_PREFIX}响应不是合法 JSON / response was not valid JSON ({exc}). "
            f"这通常说明 base_url 指向的不是 OpenAI 兼容端点。原始响应 / raw body: {_snippet(body_text)}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"{_ERROR_PREFIX}响应顶层不是对象 / response was not a JSON object. 原始响应 / raw body: {_snippet(body_text)}"
        )

    error = data.get("error")
    if error:
        # Some endpoints answer 200 with an error object in the body, so this
        # has to be checked even on the success path.
        raise ValueError(f"{_ERROR_PREFIX}endpoint 返回错误 / endpoint returned an error: {_error_message(error)}")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(
            f"{_ERROR_PREFIX}响应里没有 choices / response contained no choices. "
            f"原始响应 / raw body: {_snippet(body_text)}"
        )

    choice = choices[0]
    if not isinstance(choice, dict):
        raise ValueError(f"{_ERROR_PREFIX}choices[0] 不是对象 / choices[0] was not an object.")

    finish_reason = choice.get("finish_reason")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError(
            f"{_ERROR_PREFIX}choices[0] 里没有 message 对象 / choices[0] carried no message object "
            f"(finish_reason={finish_reason!r}). 原始响应 / raw body: {_snippet(body_text)}"
        )

    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        # Some multimodal endpoints return a list of content blocks on the way
        # back too. Flatten the text ones instead of stringifying the repr.
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            content = str(content)

    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal.strip() and not content.strip():
        raise ValueError(f"{_ERROR_PREFIX}模型拒答 / model refused to respond: {refusal.strip()}")

    return content, _reasoning_from_message(choice)

async def _post_chat(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float,
    max_retries: int,
) -> str:
    """POST the payload, retrying only transient failures, and return the body text.

    A fresh ``AsyncClient`` per call, closed by the context manager. A
    module-level client would be cheaper, but it binds to the event loop it is
    first used on, and ComfyUI is free to run node execution on a different loop
    (or a thread with its own loop) after a restart or under ``--async-offload``.
    Connection setup is negligible next to an LLM call.
    """
    attempts = max_retries + 1
    last_error: str | None = None

    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            last_error = (
                f"请求超时 / request timed out after {timeout:g}s ({type(exc).__name__}). "
                "推理模型和长上下文可能需要更长时间，可以调大 timeout。"
            )
        except httpx.HTTPError as exc:
            last_error = (
                f"连接失败 / could not reach {url} ({type(exc).__name__}: {exc}). "
                "检查 base_url 是否能从这台机器访问，以及是否需要代理。"
                "如果设了 HTTP_PROXY/HTTPS_PROXY，本地端点可能被代理拦下 —— 把 127.0.0.1 加进 NO_PROXY。"
            )
        else:
            if response.is_success:
                return response.text
            # A rewritten redirect explains the failure better than the status
            # code it ended on, so it wins when both apply.
            hint = _redirect_note(response) or _status_hint(response.status_code)
            last_error = (
                f"HTTP {response.status_code} {response.reason_phrase} 来自 / from {url}. "
                + (f"{hint} " if hint else "")
                + f"响应 / response: {_snippet(response.text)}"
            )
            if response.status_code not in _RETRYABLE_STATUS:
                break

        if attempt + 1 < attempts:
            # Exponential backoff: 1s, 2s, 4s... A 429 or an overloaded endpoint
            # usually clears well inside that.
            await asyncio.sleep(_RETRY_BASE_DELAY * (2**attempt))

    retried = f" （已尝试 {attempts} 次 / tried {attempts} times）" if attempts > 1 else ""
    raise ValueError(f"{_ERROR_PREFIX}{last_error}{retried}")
