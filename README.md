# ComfyUI-LLMApi

One node, `LLM API Chat`: point it at any OpenAI-compatible chat endpoint with
your own base URL and API key, and it posts there directly. No proxy, no Comfy
Org account, no credits.

Works against OpenAI, DeepSeek, Moonshot, Zhipu, DashScope, a vLLM or Ollama box
on your LAN, or a relay — anything that speaks the OpenAI chat-completions shape.

## Install

Drop the folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

There is nothing to install. The pack needs only what ComfyUI already ships
(`httpx`, `torch`, `PIL`, `numpy`), and `requirements.txt` is deliberately
comment-only — see [Why no dependencies](#why-no-dependencies).

Note that `httpx` arrives **transitively** (ComfyUI lists `aiohttp`, not `httpx`;
it comes in via `transformers` → `huggingface_hub`). The self-test imports it
explicitly so that if that chain ever changes, you get a named error instead of a
pack that silently disappears.

## Usage

Add **LLM API Chat** from the `text/llm api` category and fill in three fields:

| field | example |
|---|---|
| `base_url` | `https://api.deepseek.com/v1` |
| `api_key` | your key, or empty for a local server |
| `model` | `deepseek-chat` |

`/chat/completions` is appended to `base_url` automatically unless it is already
there. A missing `/v1` is **not** guessed — relays differ, and a wrong guess
produces a 404 that looks like a server problem rather than a config typo.

Two outputs: `text` (the reply) and `reasoning` (the thinking trace, empty for
models that do not produce one — safe to wire unconditionally).

Wire one or more images into the `images` slots for vision models. Each is
downscaled to fit 1280×1280 and inlined as a PNG data URI, so there is no upload
step and no external URL in the request.

### Provider examples

| provider | `base_url` | model |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Moonshot / Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
| Zhipu / GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-plus` |
| DashScope / 通义 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| Ollama (local) | `http://127.0.0.1:11434/v1` | `llama3.2-vision` |
| vLLM (local) | `http://127.0.0.1:8000/v1` | whatever you served |

Local endpoints need no key: leave `api_key` empty and the `Authorization` header
is omitted entirely.

## The API key is not secret

ComfyUI has no password widget, so the key is an ordinary text field. It is
visible on screen and is written verbatim into the workflow JSON.

Do not commit workflows containing a key, and do not share screenshots of this
node with the key filled in. To keep a key out of the workflow entirely, leave
`api_key` empty and run a local proxy in front of the endpoint that injects the
header itself.

## Cancellation

`execute` is async, so a long request does not freeze the UI — but pressing
Cancel does not abort the in-flight HTTP call. It takes effect when the response
arrives. Set `timeout` accordingly; the default is 300 s, which a long reasoning
trace can exceed.

## Why no dependencies

An HTTP client is the only thing this pack needs, and ComfyUI already depends on
`httpx`. So `requirements.txt` lists nothing, `pyproject.toml` reads its
dependency list from that same file, and a restart is the whole install.

This matters more than usual in an environment where ComfyUI-Manager has been
removed: its startup installer once ran `uv pip install` while ComfyUI held
Pillow's `_imaging` .pyd memory-mapped, so uv could not unlink the file, aborted
the transaction, and left ComfyUI unable to `import PIL.Image`. Every dependency a
pack declares is another chance at that. `_selftest.py` asserts the file stays
comment-only.

## Troubleshooting

**Connection refused against a server that is obviously running.** `httpx`
honours `HTTP_PROXY` / `HTTPS_PROXY`, so a proxy that does not exclude loopback
will intercept `http://127.0.0.1:11434/v1`. Add `127.0.0.1` and `localhost` to
`NO_PROXY`.

**A relay that redirects.** 307/308 are followed and work. 301/302/303 cannot —
they rewrite the POST into a GET and drop the body, so the error says so and
tells you to use the redirect target as `base_url`.

**Chinese UI shows English on the image slots.** That is the one tooltip a locale
file cannot override; it is written bilingually in the schema on purpose. The rest
of the node is translated in `locales/zh/`.

Remember `GET /i18n` is cached for the life of the process — editing locale JSON
requires a ComfyUI **restart**, not a page refresh.

## Verifying

```
E:\ComfyUI\.venv\Scripts\python.exe E:\ComfyUI\custom_nodes\ComfyUI-LLMApi\_selftest.py
```

No ComfyUI server and no network required. The suite compiles every module, loads
the pack the way ComfyUI's loader does, asserts the schema and the Chinese
translations match it key-for-key, and drives the full request/response path
against a throwaway `http.server` on loopback — including the wire format of
every request, all the reasoning-field variants, and every error branch.

That last part is the point: the wire format is what is most likely to be
silently wrong, and it is the one thing a hand test in the UI will not show you.

## Layout

```
ComfyUI-LLMApi/
├── __init__.py          V3 entry point (no NODE_CLASS_MAPPINGS -- see its docstring)
├── nodes/
│   ├── chat.py          LlmApiChat: schema + execute, orchestration only
│   ├── _client.py       URL joining, request shaping, HTTP, error normalisation
│   └── _media.py        image -> data URI, reasoning extraction
├── locales/zh/          node and widget translations
├── docs/LlmApiChat.md   per-node reference
└── _selftest.py         offline verification
```

## License

MIT — see [LICENSE](LICENSE).
