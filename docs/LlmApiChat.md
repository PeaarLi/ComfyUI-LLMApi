# `LlmApiChat`

**Category:** `text/llm api`
**Output kind:** text generation (one request per execution)

Send a system+user chat request to any OpenAI-compatible endpoint, using the
base URL and API key you supply. No proxy in between.

```
prompt, model, base_url, api_key, system_prompt, images, seed,
temperature, max_tokens, timeout, max_retries
  -> text (STRING), reasoning (STRING)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `prompt` | STRING (multiline) | `""` | — | The user message. May be empty when an image is wired, but `prompt` and `images` cannot both be empty. |
| `model` | STRING | `""` | — | Required. Free text, not a dropdown — see [Why `model` is free text](#why-model-is-free-text). |
| `base_url` | STRING | `https://api.openai.com/v1` | — | Endpoint prefix. `/chat/completions` is appended unless already present. |
| `api_key` | STRING | `""` | — | Bearer token. Empty sends **no** `Authorization` header at all. Stored in plain text — see [The key is not secret](#the-key-is-not-secret). |
| `system_prompt` | STRING (multiline) | `""` | — | Blank omits the system message entirely. |
| `images` | AUTOGROW(IMAGE) | no slots | up to 8 | Slot order is preserved. A batched input expands one image per frame. |
| `seed` | INT | `0` | 0–2147483647 | Sent only when > 0. Primarily a ComfyUI-side re-run trigger. |
| `temperature` | FLOAT (advanced) | `-1.0` | -1.0–2.0 | `-1` omits the field. `0` is a real value and is sent. |
| `max_tokens` | INT (advanced) | `0` | 0–1000000 | `0` omits the field. |
| `timeout` | FLOAT (advanced) | `300.0` | 1–3600 | Seconds, for the whole request. |
| `max_retries` | INT (advanced) | `0` | 0–10 | Retries connection failures, timeouts, 429 and 5xx, with exponential backoff. |

## Outputs

| display name | type | notes |
|---|---|---|
| `text` | STRING | The model's reply. Empty when the model produced only a reasoning trace. |
| `reasoning` | STRING | The thinking trace, when the endpoint returns one. Empty for models that do not produce one, so this port is always safe to wire. |

## URL joining

`base_url` has `/chat/completions` appended unless it already ends with it:

| `base_url` | request URL |
|---|---|
| `https://api.openai.com/v1` | `https://api.openai.com/v1/chat/completions` |
| `https://api.deepseek.com/v1/` | `https://api.deepseek.com/v1/chat/completions` |
| `https://relay.example.com/v1/chat/completions` | unchanged |
| `http://127.0.0.1:11434/v1` | `http://127.0.0.1:11434/v1/chat/completions` |

A missing `/v1` is **not** guessed. Relays put the version segment in different
places (`/v1`, `/openai/v1`, `/compatible-mode/v1`) or omit it, and a wrong guess
produces a 404 that reads like a server fault rather than a typo in your config.

## Why `model` is free text

ComfyUI's built-in partner nodes use a curated `DynamicCombo` because they ship a
known model list with per-model pricing and per-model widgets. This node's whole
purpose is the opposite: reaching endpoints whose model list nobody has
enumerated — a LAN vLLM serving whatever was pulled last week, a relay fronting
several vendors. There is no list to offer, so the field is a text box.

## The key is not secret

`io.String.Input` has no password variant, so the API key is an ordinary text
widget. It is visible on screen and is written verbatim into the workflow JSON.
`/object_info` and any shared screenshot expose it.

Practical consequences: do not commit workflows that carry a key, and do not
paste screenshots of this node with the key filled in. If you need the key out of
the workflow, leave `api_key` empty and put a local proxy in front of the
endpoint that injects the header itself.

An empty key omits the `Authorization` header rather than sending
`Bearer ` with nothing after it — local servers (Ollama, vLLM, LM Studio) need no
key, and an empty-but-present header is a 401 on some of them.

## Vision input

Each wired image is downscaled to fit 1280×1280, encoded as PNG and inlined as a
`data:` URI, so the request carries no external URLs and needs no upload step.
Media blocks precede the text block, matching ComfyUI core's OpenRouter node.

Two consequences worth knowing:

- **Payload size.** Base64 adds ~33% over PNG, which is already uncompressed for
  photographic input. The 1280×1280 cap exists to keep a request in the low
  megabytes; raising it multiplies the body size.
- **The model must support images.** A text-only model rejects the request with a
 4xx, and the error message it returns is the one you will see.

Batch inputs expand one image per frame, and the 8-image cap counts expanded
images, not slots.

## Redirects

Redirects are followed, and the two classes behave differently:

- **307 / 308** preserve the request method and body, so a relay that uses one
  works transparently.
- **301 / 302 / 303** instruct every compliant client to re-issue the request as
  a **GET** and drop the body (RFC 9110). A chat-completions call that arrives as
  a bodyless GET cannot succeed, and what comes back — 405, 501, or a login page
  — describes the destination rather than the actual problem. When this happens
  the error names the redirect and its target instead of blaming the endpoint for
  the status it happened to return.

The fix for the second case is always the same: point `base_url` at the final URL
the redirect was heading for.

## Reasoning extraction

The field name is a vendor extension — OpenAI's own spec does not define one — so
several are probed in order, first on `message` and then hoisted onto the choice:

1. `reasoning_content` — DeepSeek, and the many relays fronting it
2. `reasoning` — OpenRouter's normalised form
3. `reasoning_details` — OpenRouter's structured form; the text fragments are
   concatenated in order

Absence is normal, not an error. Every plain chat model returns nothing here, and
the port yields `""`.

## Errors

All failures raise, so they appear in ComfyUI's error box, and all are prefixed
`LlmApiChat: `:

| condition | what you get |
|---|---|
| Empty or whitespace `base_url` | configuration error, before any request |
| Empty `model` | configuration error |
| `prompt` and `images` both empty | configuration error |
| Connection refused / DNS failure | the URL that could not be reached, plus the httpx exception, plus a note about `HTTP_PROXY` |
| Timeout | the timeout value, and a hint that reasoning models need longer |
| Non-2xx | status, a status-specific hint (below), and the first 600 characters of the body |
| Followed 301/302/303 | the redirect and its target, and that the POST was rewritten to a GET |
| Body that is not JSON | a note that the URL probably is not an OpenAI-compatible endpoint, plus the raw body |
| `{"error": {...}}` inside a 200 | the error's `message` and `code` |
| Empty `choices`, or `choices[0]` with no `message` | the raw body, truncated |
| `refusal` with empty content | the refusal text |

Only transient conditions (connection failure, timeout, 408/409/425/429/5xx) are
retried. 401, 404 and 422 are configuration errors — retrying them would delay
the message you actually need.

Because you supply the URL, the key and the model by hand, the dominant failure
mode is a misconfiguration rather than a broken server, so the common statuses
carry their fix inline: `400` lists the usual causes, `401` points at a key and
provider that do not match, `404` mentions the missing version segment, `413`
suggests fewer or smaller images, `429` suggests raising `max_retries`.

## Behind a proxy

`httpx` honours `HTTP_PROXY` / `HTTPS_PROXY`, which is what you want when a relay
is only reachable through a proxy — but it also means a proxy that does not
exclude loopback will swallow requests to `http://127.0.0.1:11434/v1`, and the
symptom is a connection failure against a server you can plainly see running. Add
`127.0.0.1` (and `localhost`) to `NO_PROXY` if you hit that.

## Image slot tooltips

The image slots are a single `Autogrow` input, and the frontend resolves each
child slot as `images_image_1`. There is deliberately **no** such key in
`locales/zh/nodeDefs.json`: the frontend falls back to the backend string when a
locale key is missing, and the backend string comes from the template's inner
`IO.Image.Input`. So that one tooltip is written bilingually in `chat.py` — it is
the only schema string in this pack that a locale file cannot override.
`_selftest.py` asserts it stays non-empty, because an empty one shows up as a
blank tooltip on every slot in every language.

## Cancellation

`execute` is async and awaits `httpx`, so a long request does not block
ComfyUI's event loop and the UI stays responsive. But pressing Cancel mid-request
does **not** abort the in-flight HTTP call; it takes effect once the response
arrives. Size `timeout` with that in mind.

A fresh `AsyncClient` is created per call. A module-level client would be
cheaper, but it binds to the event loop it first runs on, and ComfyUI may execute
nodes on a different loop after a restart or under `--async-offload`.

## API use

```json
{
  "prompt": "Describe this in five words.",
  "model": "qwen-vl-max",
  "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "api_key": "sk-...",
  "images": {"image_1": "<IMAGE>"},
  "temperature": -1.0,
  "max_tokens": 0
}
```

`temperature: -1.0` and `max_tokens: 0` are the "omit this field" sentinels. Send
them explicitly if you are writing the node's inputs by hand — omitting them
entirely also works, since they are the schema defaults.
