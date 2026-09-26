"""Shared HTTP transport for both provider nodes.

The retry policy, the timeout mapping, the redirect handling and the shape of a
non-2xx error message live here so the OpenAI and Anthropic nodes cannot drift
apart in behaviour. What stays with each provider is the part that is genuinely
theirs: the error-message prefix, and the hint attached to a given status code
(a 404 means "your base_url is missing a version segment" for one and something
else for the other).

This was extracted rather than duplicated for one concrete reason: the redirect
handling below is subtle enough that it was written wrong the first time, and two
copies would mean two places to fix it. ``chat.py`` is unaffected -- ``_client.py``
still exposes the same ``_post_chat`` it always did.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
from typing import Any, Callable
from urllib.parse import urlparse

import httpx

__all__ = [
    "_RETRYABLE_STATUS",
    "_is_loopback",
    "_post_json",
    "_redirect_note",
    "_snippet",
]

# Hostnames that mean "this machine" but are not IP literals.
_LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})

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


def _is_loopback(url: str) -> bool:
    """Whether ``url`` targets this machine.

    Used to decide whether to let httpx honour ambient proxy settings, and the
    reason is not hypothetical. On Windows, httpx resolves proxies through
    ``urllib.request.getproxies()``, which falls back to the registry when no
    ``HTTP_PROXY`` is set -- and httpx then uses that proxy **without** the
    registry's own ``ProxyOverride`` bypass list. So with Clash (or any tool that
    enables the Windows system proxy) running, a request to ``127.0.0.1`` is
    handed to the proxy even though the bypass list says ``localhost;127.*``.

    The symptom is nasty because it is not a clean failure: with the proxy up,
    every local call pays its latency; with the proxy down but the registry still
    set, a plainly-running local server reports as unreachable.
    """
    host = urlparse(url).hostname or ""
    if not host:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host.lower() in _LOOPBACK_NAMES
    # is_unspecified covers 0.0.0.0 and ::, which the ipaddress module does not call
    # loopback. As a *destination* they mean this machine -- 0.0.0.0:8000 is a
    # common way to write "the server I just started" -- and no remote host is ever
    # 0.0.0.0, so treating them as local is both correct and safe.
    return address.is_loopback or address.is_unspecified


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "…"


def _redirect_note(response: httpx.Response) -> str:
    """Name a followed redirect that rewrote POST into GET.

    Redirects ARE followed, because 307/308 preserve the method and body and a
    relay using one is genuinely rescued by that. But 301/302/303 tell every
    compliant client to re-issue the request as a GET and drop the body, so a
    completion call arriving that way cannot succeed. What comes back -- 405,
    501, or a login page -- describes the destination, not the problem, and a
    status hint would blame the wrong party. The chain is still on
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


async def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float,
    max_retries: int,
    error_prefix: str,
    status_hint: Callable[[int], str] | None = None,
) -> str:
    """POST a JSON payload, retrying only transient failures; return the body text.

    A fresh ``AsyncClient`` per call, closed by the context manager. A
    module-level client would be cheaper, but it binds to the event loop it is
    first used on, and ComfyUI is free to run node execution on a different loop
    (or a thread with its own loop) after a restart or under ``--async-offload``.
    Connection setup is negligible next to an LLM call.

    ``follow_redirects`` is on for the 307/308 case; see ``_redirect_note`` for
    why a 301/302/303 that survives to the error path is reported specially.
    """
    attempts = max_retries + 1
    last_error: str | None = None

    client_kwargs: dict[str, Any] = {"timeout": timeout, "follow_redirects": True}
    if _is_loopback(url):
        # Never proxy a request to this machine -- see _is_loopback. trust_env also
        # switches off env-provided CA bundles, which is irrelevant for a loopback
        # call and is the price of being able to turn the proxy off per request.
        client_kwargs["trust_env"] = False

    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
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
            hint = _redirect_note(response) or (status_hint(response.status_code) if status_hint else "")
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
    raise ValueError(f"{error_prefix}{last_error}{retried}")
