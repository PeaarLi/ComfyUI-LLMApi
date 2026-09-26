# `LlmApiAnthropic`

**Category:** `text/llm api`
**Output kind:** text generation (one request per execution)

Send a system+user request to any Anthropic Messages API endpoint, using the base
URL and key you supply, with optional Agent Skills.

```
prompt, model, base_url, api_key, auth_style, anthropic_version, system_prompt,
skill, images, reasoning_trace, native_skills, max_tokens, temperature, timeout,
max_retries
  -> text (STRING), reasoning (STRING)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `prompt` | STRING (multiline) | `""` | — | The user message. May be empty when an image is wired, but not when there are none. |
| `model` | STRING | `claude-opus-5` | — | Required. Free text — a relay's model names cannot be known in advance. |
| `base_url` | STRING | `https://api.anthropic.com` | — | See [URL resolution](#url-resolution). |
| `api_key` | STRING | `""` | — | Empty sends no auth header at all. Stored in plain text — see [The key is not secret](#the-key-is-not-secret). |
| `auth_style` | COMBO (advanced) | `x-api-key` | `x-api-key` / `Authorization: Bearer` | Which header carries the key. See [Auth](#auth). |
| `anthropic_version` | STRING (advanced) | `2023-06-01` | — | Blank omits the `anthropic-version` header. |
| `system_prompt` | STRING (multiline) | `""` | — | Sent as the top-level `system` parameter, after the selected skill's text. |
| `skill` | COMBO | `(none)` | `(none)` + bundled skills | A local `SKILL.md`, injected into `system`. See [Skills](#skills). |
| `images` | AUTOGROW(IMAGE) | no slots | up to 8 | Sent as Anthropic base64 image blocks. A batch expands one image per frame. |
| `reasoning_trace` | BOOLEAN (advanced) | `True` | — | Sets `thinking.display` to `summarized`. |
| `native_skills` | STRING (advanced) | `""` | ≤ 8 ids | Anthropic-native skill ids. See [Native skills](#native-skills). |
| `max_tokens` | INT | `16000` | 1–128000 | **Required by Anthropic** — no "omit it" option. |
| `temperature` | FLOAT (advanced) | `-1.0` | -1.0–1.0 | `-1` omits the field. Range is 0–1, not the OpenAI node's 0–2. |
| `timeout` | FLOAT (advanced) | `300.0` | 1–3600 | Seconds, for the whole request. |
| `max_retries` | INT (advanced) | `0` | 0–10 | Retries connection failures, timeouts, 429 and 5xx, with backoff. |

## Outputs

| display name | type | notes |
|---|---|---|
| `text` | STRING | The concatenated `text` blocks. Empty when the model returned only a trace. |
| `reasoning` | STRING | The `thinking` blocks, joined by a blank line. Empty when there are none. |

## URL resolution

| `base_url` | request URL |
|---|---|
| `https://api.anthropic.com` | `https://api.anthropic.com/v1/messages` |
| `https://api.anthropic.com/v1` | `https://api.anthropic.com/v1/messages` |
| `https://api.anthropic.com/v1/messages` | unchanged |
| `https://relay.example.com/anthropic` | `https://relay.example.com/anthropic/v1/messages` |
| `https://relay.example.com/messages` | unchanged |

Both the bare host (which Anthropic's own docs use) and a host ending in `/v1`
(which the sibling `LlmApiChat` node trains you to type) resolve to the same
place — appending `/v1/messages` blindly would turn the second into
`/v1/v1/messages`. Pasting a complete URL that ends in `/messages` always works
and is the escape hatch when a relay's paths differ.

## Auth

Two styles, and only one header is **ever** sent:

| `auth_style` | header |
|---|---|
| `x-api-key` (default) | `x-api-key: <key>` |
| `Authorization: Bearer` | `Authorization: Bearer <key>` |

Official Anthropic wants the first. A large share of relays only implement the
second, because that is what every other OpenAI-compatible client sends. Sending
both is not a safe middle ground — gateways that reject unexpected headers exist —
so this is an explicit choice, and picking wrong shows up as a 401.

An empty `api_key` omits credentials entirely rather than sending an empty value,
which is what lets a local Anthropic-compatible server run without one.

`anthropic_version` is sent as the `anthropic-version` header; blank omits it for
endpoints that reject it.

## Skills

Two independent routes. They are not interchangeable, and the difference is the
whole reason both exist.

### Local (`skill`)

A `SKILL.md` from this pack's `skills/` directory. The frontmatter is parsed for
its required `name` and `description`, and the **body** is injected into the
`system` parameter, before your own `system_prompt` — so anything you write there
overrides the skill.

- Works against **any** Anthropic-shaped endpoint, including relays, because it is
  only text.
- **It is an approximation of the real mechanism.** Anthropic's own container route
  lets the model read supporting files on demand; here the body is all the model
  gets, so a skill that depends on `references/` or `scripts/` will not work.
- The combo's list is built **at import**, so adding or removing a skill directory
  needs a ComfyUI restart. Editing an existing skill's *text* does not — bodies are
  re-read on every run.
- A malformed manifest is skipped at discovery with a log warning rather than
  taken down the whole pack; loading a skill you actually selected raises.

The format is the open Agent Skills standard, which means these same files work in
Claude Code, Codex, Gemini CLI and Copilot CLI. Copying `skills/<name>/` to
`~/.agents/skills/` is all it takes — that directory is the cross-runtime alias
those tools share.

### Native (`native_skills`)

Skill ids attached as `container.skills` together with the code-execution tool, so
the model reads them from a server-side sandbox. This is the real mechanism.

- Ids are comma- or newline-separated, **max 8 per request** (checked locally).
- Anthropic's prebuilt document skills are `xlsx`, `docx`, `pptx`, `pdf`. Those
  four are sent with type `anthropic`; anything else is sent as `custom`.
- **A custom skill has to be published first** — there is no inline form on
  Anthropic's side, unlike OpenAI's. Local files cannot be used directly:

```bash
# Publish skills/<name>/ and note the skill_id it returns.
curl -X POST 'https://api.anthropic.com/v1/skills' \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: skills-2025-10-02" \
  -F 'files[]=@skills/image-prompt-enhancer/SKILL.md;filename=image-prompt-enhancer/SKILL.md;type=text/markdown'

# List what is available, prebuilt and custom.
curl 'https://api.anthropic.com/v1/skills' -H "x-api-key: $ANTHROPIC_API_KEY"
```

Constraints worth knowing before you plan around this route: skill bundles are
capped at 30 MB, and **the container has no network access and cannot install
packages** — a skill that shells out to `pip install` or calls an external API
works in Claude Code and fails here.

**The container is Anthropic's own infrastructure, so a relay will usually fail
here.** That is not blocked on the host — a relay that passes the container
through does work — but when this path fails with a 400 or 404 the error says so
and points you back at the local `skill` dropdown, rather than passing on a bare
status code.

Both routes may be set at once. They are independent.

## Reasoning

`reasoning_trace` on (the default) sends
`thinking: {"type": "adaptive", "display": "summarized"}`.

The `display` half is the part that matters. Thinking runs by default on the
current models, but its text comes back **empty** unless a summarized display is
requested — so without this the `reasoning` output would always be blank, which
reads as a bug rather than a default.

Turning it off omits the parameter entirely. Thinking still happens; its text just
is not returned, so you save only what the summary itself would have cost.

## Request shape differences that bite

The Messages API is not chat-completions with a different URL. The four that are
easiest to get wrong, all handled by this node:

- **Images** are `{"type": "image", "source": {"type": "base64", "media_type": ..., "data": ...}}`,
  not a `data:` URI under `image_url`.
- **`system` is top-level**, not a message with `role: "system"`.
- **`max_tokens` is required**, so unlike the sibling node there is no
  `0 = omit` sentinel; the default is a real 16000.
- **The response is a `content` block array**, not `choices`, and the error
  envelope nests under a `type` key.

## Errors

All failures raise, prefixed `LlmApiAnthropic: `.

| condition | what you get |
|---|---|
| Empty `model` | configuration error |
| `max_tokens` below 1 | configuration error, naming Anthropic's requirement |
| `prompt` and `images` both empty | configuration error |
| More than 8 `native_skills` | configuration error, before any request |
| Unknown `skill` | the path it looked in, and what is available |
| Connection refused / DNS failure | the URL, the httpx exception, and a `HTTP_PROXY` note |
| Timeout | the timeout value |
| Non-2xx | status, a status hint, and the first 600 characters of the body |
| Body that is not JSON | a note that the URL probably is not an Anthropic endpoint, plus the raw body |
| An OpenAI-shaped body | the top-level keys it did find, and that this endpoint returns `content`, not `choices` |
| `stop_reason: "refusal"` | raises, with `stop_details.category` |
| `stop_reason: "max_tokens"` | **not** fatal — the partial text is returned with a log warning |
| 400/404 **with** native skills | the container prerequisite, and a pointer to the local `skill` route |

Only transient conditions (connection failure, timeout, 408/409/425/429/5xx) are
retried. 401, 404 and 422 are configuration errors, and retrying them only delays
the message you need.

## The key is not secret

ComfyUI has no password widget, so the API key is an ordinary text field, visible
on screen and written verbatim into the workflow JSON. Do not commit workflows
containing a key and do not share screenshots of this node with the field filled
in. To keep a key out of the workflow, leave `api_key` empty and put a local proxy
in front of the endpoint that injects the header itself.

## Cancellation

`execute` is async, so a long request does not freeze the UI — but pressing Cancel
does not abort the in-flight HTTP call. It takes effect when the response arrives.
On the native-skill path the container adds its own latency on top, so size
`timeout` generously.

## API use

```json
{
  "prompt": "Expand this into an image prompt: a lighthouse",
  "model": "claude-opus-5",
  "base_url": "https://api.anthropic.com",
  "api_key": "sk-ant-...",
  "skill": "image-prompt-enhancer",
  "system_prompt": "",
  "reasoning_trace": false,
  "max_tokens": 16000,
  "temperature": -1.0,
  "native_skills": ""
}
```

`temperature: -1.0` is the "omit this field" sentinel. `reasoning_trace: false`
and empty strings are also the schema defaults, so omitting them entirely works
too.

**Only `prompt` and `model` are required** in an API-submitted prompt. Every
other input is declared `optional`, so you can send just the fields you care
about and ComfyUI fills the rest from the schema defaults — the JSON above is a
full example, not a minimum. (ComfyUI rejects a prompt that omits a *required*
input, which is why the split is deliberate rather than uniform.)

Note that `skill` is a combo whose value is the skill's **directory name**; the
`(none)` sentinel is how you run without one.
