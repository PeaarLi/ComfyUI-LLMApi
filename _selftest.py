"""Standalone verification for ComfyUI-LLMApi. No ComfyUI server, no network.

Run with the ComfyUI interpreter::

    E:\\ComfyUI\\.venv\\Scripts\\python.exe E:\\ComfyUI\\custom_nodes\\ComfyUI-LLMApi\\_selftest.py

Why this file exists rather than a pytest suite: importing a node pack is where
things go wrong quietly. An exception raised inside ``define_schema`` -- a kwarg
the schema does not accept, a bad enum value -- is swallowed by ComfyUI's loader
into a single log warning and the entire pack then disappears from the menu with
no error dialog. Nothing in ComfyUI tells you which node broke. So the first
thing section_schema does is import the pack the same way the loader does and
call ``GET_SCHEMA()`` on every node, which turns that silent disappearance into a
traceback.

The request/response path is exercised against a throwaway ``http.server`` on a
loopback port, so the whole thing runs offline and needs no API key. That is what
makes it safe to assert on the exact bytes the node sends -- the wire format is
the part most likely to be silently wrong.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import io as io_module
import json
import py_compile
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
COMFY_ROOT = PACK_DIR.parent.parent
PACK_NAME = "llm_api_chat_pack"

for path in (str(COMFY_ROOT), str(PACK_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402  (after sys.path setup)
from PIL import Image  # noqa: E402

_FAILURES: list[str] = []
_PASSES = 0


def check(name: str, condition: bool, detail: str = "") -> bool:
    global _PASSES
    if condition:
        _PASSES += 1
        print(f"  ok   {name}")
    else:
        _FAILURES.append(name)
        print(f"  FAIL {name}" + (f"\n       {detail}" if detail else ""))
    return bool(condition)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def load_pack():
    """Import the pack exactly the way ``nodes.py load_custom_node`` does.

    The directory name ``ComfyUI-LLMApi`` is not a valid Python identifier, so
    the package cannot be imported by name; ``spec_from_file_location`` with
    ``submodule_search_locations`` is how you get a real package (relative
    imports intact) out of a hyphenated folder.
    """
    spec = importlib.util.spec_from_file_location(
        PACK_NAME, PACK_DIR / "__init__.py", submodule_search_locations=[str(PACK_DIR)]
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACK_NAME] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def section_compile() -> None:
    section("py_compile")
    sources = sorted(PACK_DIR.rglob("*.py"))
    check("found python sources", len(sources) >= 5, f"found {len(sources)}")
    for source in sources:
        try:
            py_compile.compile(str(source), doraise=True, cfile=str(source) + ".pyc-check")
        except py_compile.PyCompileError as exc:
            check(f"compiles: {source.relative_to(PACK_DIR)}", False, str(exc))
        else:
            check(f"compiles: {source.relative_to(PACK_DIR)}", True)
        finally:
            Path(str(source) + ".pyc-check").unlink(missing_ok=True)


def section_pack(pack) -> type:
    section("pack import / registration")
    init_source = (PACK_DIR / "__init__.py").read_text(encoding="utf-8")
    # The single most destructive mistake available in this pack: defining
    # NODE_CLASS_MAPPINGS makes the loader take the V1 branch and skip
    # comfy_entrypoint entirely, so the pack vanishes silently. Checked with
    # hasattr rather than a source grep, because that is literally the test
    # load_custom_node performs -- and a docstring merely *mentioning* the name
    # (as this pack's does, to warn against it) must not trip the check.
    check(
        "__init__.py does not define NODE_CLASS_MAPPINGS",
        not hasattr(pack, "NODE_CLASS_MAPPINGS"),
        "defining it routes load_custom_node down the V1 path and skips comfy_entrypoint",
    )
    check("__init__.py still documents the V1/V3 trap", "NODE_CLASS_MAPPINGS" in init_source)
    check("pack exposes comfy_entrypoint", callable(getattr(pack, "comfy_entrypoint", None)))
    check("ALL_NODES is non-empty", bool(getattr(pack, "ALL_NODES", None)))

    nodes = list(pack.ALL_NODES)
    check("ALL_NODES has exactly one node", len(nodes) == 1, f"got {[n.__name__ for n in nodes]}")

    # GET_SCHEMA is the call the loader makes (nodes.py: for node_cls in node_list:
    # schema = node_cls.GET_SCHEMA()). Calling it here surfaces import-time
    # failures ComfyUI would swallow into a warning.
    for node_cls in nodes:
        try:
            schema = node_cls.GET_SCHEMA()
        except Exception as exc:  # noqa: BLE001 - the point is to catch everything
            check(f"{node_cls.__name__}.GET_SCHEMA()", False, f"{type(exc).__name__}: {exc}")
        else:
            check(f"{node_cls.__name__}.GET_SCHEMA()", schema is not None)
    return nodes[0]


def section_schema(node_cls) -> None:
    section("schema")
    schema = node_cls.GET_SCHEMA()
    check("node_id", schema.node_id == "LlmApiChat", f"got {schema.node_id!r}")
    check("display_name", schema.display_name == "LLM API Chat", f"got {schema.display_name!r}")
    check("category", schema.category == "text/llm api", f"got {schema.category!r}")

    expected_inputs = [
        "prompt",
        "model",
        "base_url",
        "api_key",
        "system_prompt",
        "images",
        "seed",
        "temperature",
        "max_tokens",
        "timeout",
        "max_retries",
    ]
    got_inputs = [item.id for item in schema.inputs]
    check("input ids match exactly", got_inputs == expected_inputs, f"got {got_inputs}")

    check("exactly two outputs", len(schema.outputs) == 2, f"got {len(schema.outputs)}")
    for index, output in enumerate(schema.outputs):
        # io_type is the wire type ComfyUI routes on; the Python class of the
        # Output wrapper is the same generic class for every type.
        check(f"output {index} is STRING", output.io_type == "STRING", f"got {output.io_type!r}")
        check(f"output {index} has a display_name", bool((output.display_name or "").strip()))
        check(f"output {index} has a tooltip", bool((output.tooltip or "").strip()))

    # The Autogrow image slots must not leak into schema.inputs as separate
    # entries -- they are a single dict-typed input. If this ever changes, the
    # locale assertion below and the execute signature both need revisiting.
    check(
        "images is a single input, not expanded into slots",
        "images" in got_inputs and not any(name.startswith("image_") for name in got_inputs),
        f"got {got_inputs}",
    )

    module = importlib.import_module(f"{PACK_NAME}.nodes.chat")
    check("DEFAULT_BASE_URL", module.DEFAULT_BASE_URL == "https://api.openai.com/v1")
    check("temperature sentinel is negative", module.DEFAULT_TEMPERATURE < 0)
    check("max_tokens sentinel omits the field", module.DEFAULT_MAX_TOKENS == 0)
    check("seed sentinel omits the field", module.DEFAULT_SEED == 0)
    check("MAX_IMAGES within engine limit", 0 < module.MAX_IMAGES <= 100, f"got {module.MAX_IMAGES}")


def section_autogrow(node_cls) -> None:
    section("Autogrow image slots")
    from comfy_api.latest import io

    schema = node_cls.GET_SCHEMA()
    images = next((item for item in schema.inputs if item.id == "images"), None)
    if not check("images input is present", images is not None):
        return
    if not check("images is an Autogrow input", isinstance(images, io.Autogrow.Input), f"got {type(images).__name__}"):
        return

    template = images.template
    check("template is TemplateNames", isinstance(template, io.Autogrow.TemplateNames), f"got {type(template).__name__}")
    check("template min is 0, so no slot is required", getattr(template, "min", None) == 0, f"got {getattr(template, 'min', None)}")
    names = list(getattr(template, "names", []))
    check("slot count matches IMAGE_SLOT_NAMES", names == [f"image_{i}" for i in range(1, len(names) + 1)], f"got {names}")
    check("slots are numbered from image_1", names[:1] == ["image_1"], f"got {names[:1]}")
    check("slot names are unique", len(names) == len(set(names)))

    # Load-bearing: the frontend resolves a child slot as "images_image_1" and
    # falls back to this backend string when the locale has no such key. Empty
    # here means a blank tooltip on every image slot, in every language.
    inner_tooltip = getattr(getattr(template, "input", None), "tooltip", None)
    check(
        "the template's inner input carries a tooltip",
        bool((inner_tooltip or "").strip()),
        "locales/zh cannot override this one; the frontend falls back to it for every child slot",
    )
    check(
        "the child slot tooltip is not duplicated in the locale",
        "images_image_1" not in (PACK_DIR / "locales" / "zh" / "nodeDefs.json").read_text(encoding="utf-8"),
        "a locale key here is dead: the frontend looks up images_image_1 but the backend string wins",
    )


def section_locale(node_cls) -> None:
    section("locale (zh)")
    schema = node_cls.GET_SCHEMA()
    main_path = PACK_DIR / "locales" / "zh" / "main.json"
    defs_path = PACK_DIR / "locales" / "zh" / "nodeDefs.json"
    if not check("main.json exists", main_path.is_file()) or not check("nodeDefs.json exists", defs_path.is_file()):
        return

    main = json.loads(main_path.read_text(encoding="utf-8"))
    categories = main.get("nodeCategories", {})
    check(
        "main.json translates the category",
        categories.get("llm api") not in (None, ""),
        f"nodeCategories={categories!r}; the key is the LAST segment of category='text/llm api'",
    )

    spec = json.loads(defs_path.read_text(encoding="utf-8")).get(schema.node_id)
    if not check("nodeDefs.json has an entry for the node_id", isinstance(spec, dict)):
        return

    for field in ("display_name", "description"):
        check(f"{field} is non-empty", bool(str(spec.get(field, "")).strip()), f"got {spec.get(field)!r}")

    # Exact equality, not "at least": a stale key left behind after a parameter
    # is renamed shows up as a blank tooltip in the UI and is very hard to spot.
    expected_inputs = {item.id for item in schema.inputs}
    got_inputs = set(spec.get("inputs", {}))
    check(
        "locale input keys match the schema exactly",
        got_inputs == expected_inputs,
        f"missing={sorted(expected_inputs - got_inputs)} dead={sorted(got_inputs - expected_inputs)}",
    )

    expected_outputs = {str(index) for index in range(len(schema.outputs))}
    got_outputs = set(spec.get("outputs", {}))
    check(
        "locale output keys are the positional indices",
        got_outputs == expected_outputs,
        f"missing={sorted(expected_outputs - got_outputs)} dead={sorted(got_outputs - expected_outputs)}",
    )

    for group, keys in (("inputs", expected_inputs), ("outputs", expected_outputs)):
        for key in sorted(keys & set(spec.get(group, {}))):
            entry = spec[group][key]
            check(f"{group}.{key} has a name", bool(str(entry.get("name", "")).strip()))
            check(f"{group}.{key} has a tooltip", bool(str(entry.get("tooltip", "")).strip()))


def section_dependencies() -> None:
    section("dependencies")
    # The pack declares no dependencies, so it leans entirely on what ComfyUI
    # already ships. httpx in particular is transitive (transformers ->
    # huggingface_hub -> httpx), not a direct ComfyUI requirement -- if that
    # chain ever breaks, the pack would vanish at import with only a log
    # warning, which is the worst possible way to find out. Import it here so
    # the failure is loud and named.
    for name, purpose in (
        ("httpx", "HTTP transport"),
        ("torch", "IMAGE tensors"),
        ("PIL", "PNG encoding"),
        ("numpy", "tensor -> array"),
    ):
        try:
            importlib.import_module(name)
        except ImportError as exc:
            check(f"{name} is importable ({purpose})", False, f"{exc}")
        else:
            check(f"{name} is importable ({purpose})", True)

    import httpx

    version = getattr(httpx, "__version__", "")
    check("httpx is a 0.x release", version.startswith("0."), f"got {version!r}; this pack targets the 0.x API")


def section_packaging() -> None:
    section("packaging")
    requirements = PACK_DIR / "requirements.txt"
    if check("requirements.txt exists", requirements.is_file()):
        # This pack must stay dependency-free. A stray bare line here -- even a
        # single word left behind by an edit -- makes ComfyUI Manager try to
        # install it at startup, which is the exact failure mode that got
        # Manager removed from this machine.
        active = [
            line.strip()
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        check("requirements.txt declares no dependencies", active == [], f"uncommented lines: {active}")

    pyproject = PACK_DIR / "pyproject.toml"
    if check("pyproject.toml exists", pyproject.is_file()):
        text = pyproject.read_text(encoding="utf-8")
        check(
            "pyproject wires dependencies to requirements.txt",
            'dependencies = { file = ["requirements.txt"] }' in text,
            "expected the dynamic-dependencies wiring so the two files cannot disagree",
        )
        check("pyproject carries a [tool.comfy] section", "[tool.comfy]" in text)
        check("pyproject states a tested requires-comfyui floor", "requires-comfyui" in text)

    check("LICENSE exists", (PACK_DIR / "LICENSE").is_file())
    check("README exists", (PACK_DIR / "README.md").is_file())
    check(
        "each node has a doc page",
        (PACK_DIR / "docs" / "LlmApiChat.md").is_file(),
        "expected docs/LlmApiChat.md, one per node",
    )


def section_url() -> None:
    section("URL joining")
    client = importlib.import_module(f"{PACK_NAME}.nodes._client")
    join = client._chat_completions_url
    cases = [
        ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions"),
        ("https://api.openai.com/v1/", "https://api.openai.com/v1/chat/completions"),
        ("http://127.0.0.1:11434/v1", "http://127.0.0.1:11434/v1/chat/completions"),
        (
            "https://relay.example.com/v1/chat/completions",
            "https://relay.example.com/v1/chat/completions",
        ),
        (
            "  https://relay.example.com/v1/chat/completions/  ",
            "https://relay.example.com/v1/chat/completions",
        ),
        ("https://api.deepseek.com/v1///", "https://api.deepseek.com/v1/chat/completions"),
    ]
    for raw, expected in cases:
        try:
            got = join(raw)
        except Exception as exc:  # noqa: BLE001
            check(f"join({raw!r})", False, f"raised {type(exc).__name__}: {exc}")
        else:
            check(f"join({raw!r}) -> {expected}", got == expected, f"got {got!r}")

    for blank in ("", "   ", None):
        try:
            join(blank)
        except ValueError as exc:
            check(f"join({blank!r}) raises ValueError", "LlmApiChat: " in str(exc), f"got {exc}")
        else:
            check(f"join({blank!r}) raises ValueError", False, "no exception raised")

    check("empty api_key sends no header", client._auth_headers("") == {})
    check("whitespace api_key sends no header", client._auth_headers("   ") == {})
    check("api_key becomes a Bearer header", client._auth_headers("sk-abc") == {"Authorization": "Bearer sk-abc"})
    check("api_key is stripped", client._auth_headers("  sk-abc  ") == {"Authorization": "Bearer sk-abc"})


# --------------------------------------------------------------------------- #
# Stub endpoint
# --------------------------------------------------------------------------- #


class _StubServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[dict] = []
        self._scripted: list[tuple] = []

    def script(self, responses: list[tuple]) -> None:
        """Queue ``(status, payload, delay[, headers])``; payload may be bytes."""
        self._scripted = list(responses)
        self.requests.clear()

    def next_response(self) -> tuple:
        if self._scripted:
            entry = self._scripted.pop(0)
            return (*entry[:3], entry[3] if len(entry) > 3 else {})
        return (200, {"choices": [{"message": {"content": "ok"}}]}, 0.0, {})


class _StubHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # silence stderr access logging
        pass

    def do_GET(self) -> None:
        # Only reachable when a followed redirect rewrote POST into GET. Answered
        # explicitly (rather than left to BaseHTTPRequestHandler's default 501) so
        # the redirect test asserts against a known status.
        length = 0
        self.send_response(405)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(length))
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            body = None
        # Keep the live Message object: it looks headers up case-insensitively,
        # which a plain dict() of it would not.
        self.server.requests.append({"path": self.path, "headers": self.headers, "body": body, "raw": raw})

        status, payload, delay, extra_headers = self.server.next_response()
        if delay:
            time.sleep(delay)
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status)
            for name, value in extra_headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            # Expected when the client timed out and went away mid-handler.
            pass


def section_http(node_cls, node_module) -> None:
    section("HTTP round-trip against a local stub")
    import asyncio

    client = importlib.import_module(f"{PACK_NAME}.nodes._client")
    # Keep the retry test fast: this is a module constant precisely so a test can
    # shorten it, and the backoff schedule itself is not what is under test.
    client._RETRY_BASE_DELAY = 0.01

    server = _StubServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    expected_path = "/v1/chat/completions"

    def run(**overrides):
        kwargs = {
            "prompt": "hello",
            "model": "test-model",
            "base_url": base_url,
            "api_key": "sk-test-key",
        }
        kwargs.update(overrides)
        return asyncio.run(node_cls.execute(**kwargs))

    def last_request():
        return server.requests[-1]

    try:
        # --- text only, defaults -------------------------------------------- #
        server.script([(200, {"choices": [{"message": {"content": "hi there"}}]}, 0.0)])
        text, reasoning = run()
        check("text request returns the content", text == "hi there", f"got {text!r}")
        check("missing reasoning yields an empty string", reasoning == "", f"got {reasoning!r}")

        request = last_request()
        check("posts to /chat/completions", request["path"] == expected_path, f"got {request['path']!r}")
        check(
            "sends the bearer header",
            request["headers"].get("Authorization") == "Bearer sk-test-key",
            f"got {request['headers'].get('Authorization')!r}",
        )
        body = request["body"]
        check("body has the model", body.get("model") == "test-model", f"got {body.get('model')!r}")
        check("stream is off", body.get("stream") is False, f"got {body.get('stream')!r}")
        check(
            "no system message when system_prompt is blank",
            [m["role"] for m in body["messages"]] == ["user"],
            f"got {[m['role'] for m in body['messages']]}",
        )
        check(
            "text-only content is a plain string",
            body["messages"][0]["content"] == "hello",
            f"got {body['messages'][0]['content']!r}",
        )
        for omitted in ("temperature", "max_tokens", "seed"):
            check(f"sentinel default omits {omitted}", omitted not in body, f"body has {omitted}={body.get(omitted)!r}")

        # --- system prompt + non-default sampling --------------------------- #
        server.script([(200, {"choices": [{"message": {"content": "ok"}}]}, 0.0)])
        run(system_prompt="be terse", temperature=0.0, max_tokens=256, seed=7)
        body = last_request()["body"]
        check(
            "system prompt becomes the first message",
            [m["role"] for m in body["messages"]] == ["system", "user"],
            f"got {[m['role'] for m in body['messages']]}",
        )
        check("system prompt content", body["messages"][0]["content"] == "be terse")
        # temperature=0 is a real value and must survive: this is exactly why the
        # sentinel is -1 rather than 0.
        check("temperature 0.0 is sent", body.get("temperature") == 0.0, f"got {body.get('temperature')!r}")
        check("max_tokens is sent", body.get("max_tokens") == 256, f"got {body.get('max_tokens')!r}")
        check("seed is sent", body.get("seed") == 7, f"got {body.get('seed')!r}")

        # --- redirects --------------------------------------------------------- #
        # httpx defaults follow_redirects to False, and a 3xx status is < 400, so
        # a stub redirect used to sail through as a successful response and fail
        # later on an empty body. Redirects are now followed -- 307 preserves the
        # method and body, so a relay using one is genuinely rescued.
        server.script(
            [
                (307, b"", 0.0, {"Location": expected_path}),
                (200, {"choices": [{"message": {"content": "after redirect"}}]}, 0.0),
            ]
        )
        text, _ = run()
        check("a 307 redirect is followed", text == "after redirect", f"got {text!r}")
        check("the followed request is still a POST", len(server.requests) == 2, f"got {len(server.requests)}")

        # But 301/302/303 make the client re-issue as a GET and drop the body
        # (RFC 9110), so the destination sees a bodyless GET and cannot possibly
        # answer. The extra GET lands on the stub's do_GET, which answers 405.
        # What matters is that the error names the redirect rather than blaming
        # the endpoint for whatever status the bodyless GET happened to produce.
        server.script([(302, b"", 0.0, {"Location": expected_path})])
        try:
            run()
        except ValueError as exc:
            message = str(exc)
            check("a 302 is reported as a redirect problem", "重定向" in message, f"got {message}")
            check("...and explains the POST-to-GET rewrite", "GET" in message, f"got {message}")
            check("...and does not blame the destination's status", "端点侧故障" not in message, f"got {message}")
        else:
            check("a 302 is reported as a redirect problem", False, "no exception raised")

        # --- no api key ------------------------------------------------------ #
        server.script([(200, {"choices": [{"message": {"content": "ok"}}]}, 0.0)])
        run(api_key="")
        check(
            "empty api_key omits the Authorization header entirely",
            last_request()["headers"].get("Authorization") is None,
            f"got {last_request()['headers'].get('Authorization')!r}",
        )

        # --- reasoning variants ---------------------------------------------- #
        server.script(
            [(200, {"choices": [{"message": {"content": "answer", "reasoning_content": "step by step"}}]}, 0.0)]
        )
        text, reasoning = run()
        check("reasoning_content is extracted", reasoning == "step by step", f"got {reasoning!r}")
        check("content still returned alongside reasoning", text == "answer", f"got {text!r}")

        server.script([(200, {"choices": [{"message": {"content": "answer", "reasoning": "openrouter style"}}]}, 0.0)])
        _, reasoning = run()
        check("reasoning is extracted", reasoning == "openrouter style", f"got {reasoning!r}")

        server.script([(200, {"choices": [{"reasoning_content": "hoisted", "message": {"content": "a"}}]}, 0.0)])
        _, reasoning = run()
        check("reasoning hoisted onto the choice is found", reasoning == "hoisted", f"got {reasoning!r}")

        server.script(
            [
                (
                    200,
                    {"choices": [{"message": {"content": "a", "reasoning_details": [{"text": "part1"}, {"text": "part2"}]}}]},
                    0.0,
                )
            ]
        )
        _, reasoning = run()
        check("reasoning_details are concatenated", reasoning == "part1part2", f"got {reasoning!r}")

        server.script([(200, {"choices": [{"message": {"content": "", "reasoning_content": "only thinking"}}]}, 0.0)])
        text, reasoning = run()
        check("reasoning-only response leaves text empty", text == "", f"got {text!r}")
        check("reasoning-only response still returns the trace", reasoning == "only thinking", f"got {reasoning!r}")

        server.script([(200, {"choices": [{"message": {"content": [{"type": "text", "text": "block form"}]}}]}, 0.0)])
        text, _ = run()
        check("content block list is flattened", text == "block form", f"got {text!r}")

        # --- images ----------------------------------------------------------- #
        server.script([(200, {"choices": [{"message": {"content": "seen"}}]}, 0.0)])
        run(images={"image_1": torch.zeros(1, 64, 32, 3)})
        content = last_request()["body"]["messages"][-1]["content"]
        check("vision content is a block list", isinstance(content, list), f"got {type(content).__name__}")
        check("one image block plus one text block", len(content) == 2, f"got {len(content)}")
        check("image block comes first", content[0].get("type") == "image_url", f"got {content[0].get('type')!r}")
        check("text block comes last", content[-1] == {"type": "text", "text": "hello"}, f"got {content[-1]!r}")
        uri = content[0]["image_url"]["url"]
        check("image is an inline png data uri", uri.startswith("data:image/png;base64,"), f"got {uri[:40]!r}")
        decoded = Image.open(io_module.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        check("image survives the round trip at its own size", decoded.size == (32, 64), f"got {decoded.size}")

        # Slot order must follow the declared names, not dict insertion order.
        server.script([(200, {"choices": [{"message": {"content": "seen"}}]}, 0.0)])
        run(images={"image_2": torch.ones(1, 8, 8, 3), "image_1": torch.zeros(1, 8, 8, 3)})
        content = last_request()["body"]["messages"][-1]["content"]
        check("two images plus text", len(content) == 3, f"got {len(content)}")
        first_pixels = Image.open(io_module.BytesIO(base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])))
        check(
            "image_1 is sent before image_2 despite dict order",
            first_pixels.getpixel((0, 0))[0] == 0,
            f"first block starts with {first_pixels.getpixel((0, 0))}",
        )

        # A batch expands one image per frame.
        server.script([(200, {"choices": [{"message": {"content": "seen"}}]}, 0.0)])
        run(images={"image_1": torch.zeros(3, 8, 8, 3)})
        content = last_request()["body"]["messages"][-1]["content"]
        check("batched input expands one block per frame", len(content) == 4, f"got {len(content)}")

        # Channel counts.
        for channels, expected_mode in ((1, "RGB"), (4, "RGBA")):
            server.script([(200, {"choices": [{"message": {"content": "seen"}}]}, 0.0)])
            run(images={"image_1": torch.zeros(1, 8, 8, channels)})
            uri = last_request()["body"]["messages"][-1]["content"][0]["image_url"]["url"]
            decoded = Image.open(io_module.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
            check(f"{channels}-channel image encodes as {expected_mode}", decoded.mode == expected_mode, f"got {decoded.mode}")

        # Oversized images are downscaled under the per-image budget.
        server.script([(200, {"choices": [{"message": {"content": "seen"}}]}, 0.0)])
        run(images={"image_1": torch.zeros(1, 2000, 2000, 3)})
        uri = last_request()["body"]["messages"][-1]["content"][0]["image_url"]["url"]
        decoded = Image.open(io_module.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
        pixels = decoded.size[0] * decoded.size[1]
        check("oversized image is downscaled within budget", pixels <= 1280 * 1280, f"got {decoded.size}")
        check("downscale preserves aspect ratio", abs(decoded.size[0] - decoded.size[1]) == 0, f"got {decoded.size}")

        # Undeclared slot names are ignored: the frontend only ever creates the
        # names the schema declares, so anything else is not ours to send.
        server.script([(200, {"choices": [{"message": {"content": "seen"}}]}, 0.0)])
        run(images={"image_1": torch.zeros(1, 8, 8, 3), "image_99": torch.zeros(1, 8, 8, 3)})
        check(
            "undeclared image slots are ignored",
            len(last_request()["body"]["messages"][-1]["content"]) == 2,
            f"got {len(last_request()['body']['messages'][-1]['content'])} blocks",
        )

        # The cap is only reachable through batch expansion, since the schema
        # declares exactly MAX_IMAGES slots.
        max_images = node_module.MAX_IMAGES
        try:
            run(images={"image_1": torch.zeros(max_images + 1, 8, 8, 3)})
        except ValueError as exc:
            check("more than MAX_IMAGES is rejected", "LlmApiChat: " in str(exc), f"got {exc}")
        else:
            check("more than MAX_IMAGES is rejected", False, "no exception raised")

        # ...and exactly MAX_IMAGES is still fine.
        server.script([(200, {"choices": [{"message": {"content": "seen"}}]}, 0.0)])
        run(images={"image_1": torch.zeros(max_images, 8, 8, 3)})
        check(
            "exactly MAX_IMAGES is accepted",
            len(last_request()["body"]["messages"][-1]["content"]) == max_images + 1,
            f"got {len(last_request()['body']['messages'][-1]['content'])} blocks",
        )

        # --- input validation -------------------------------------------------- #
        for name, overrides in (("blank model", {"model": "  "}), ("blank prompt and no images", {"prompt": ""})):
            try:
                run(**overrides)
            except ValueError as exc:
                check(f"{name} is rejected", "LlmApiChat: " in str(exc), f"got {exc}")
            else:
                check(f"{name} is rejected", False, "no exception raised")

        # An image with no prompt is legitimate ("describe this").
        server.script([(200, {"choices": [{"message": {"content": "described"}}]}, 0.0)])
        text, _ = run(prompt="", images={"image_1": torch.zeros(1, 8, 8, 3)})
        check("image-only request is allowed", text == "described", f"got {text!r}")

        # --- error branches ---------------------------------------------------- #
        error_cases = [
            ("401", (401, {"error": {"message": "invalid api key"}}, 0.0), "HTTP 401"),
            ("500 after retries exhausted", (500, {"error": {"message": "boom"}}, 0.0), "HTTP 500"),
            ("malformed json", (200, b"<html>not json</html>", 0.0), "not valid JSON"),
            ("error object with status 200", (200, {"error": {"message": "model not found"}}, 0.0), "model not found"),
            ("empty choices", (200, {"choices": []}, 0.0), "no choices"),
            ("missing message object", (200, {"choices": [{"finish_reason": "content_filter"}]}, 0.0), "no message object"),
            ("empty body", (200, b"", 0.0), "empty body"),
        ]
        for name, response, needle in error_cases:
            server.script([response])
            try:
                run()
            except ValueError as exc:
                message = str(exc)
                check(f"{name} raises", "LlmApiChat: " in message, f"got {message}")
                check(f"{name} message mentions the cause", needle in message, f"got {message}")
            else:
                check(f"{name} raises", False, "no exception raised")

        server.script([(200, {"choices": [{"message": {"content": "", "refusal": "I cannot help with that"}}]}, 0.0)])
        try:
            run()
        except ValueError as exc:
            check("refusal raises", "I cannot help with that" in str(exc), f"got {exc}")
        else:
            check("refusal raises", False, "no exception raised")

        # --- timeout ----------------------------------------------------------- #
        server.script([(200, {"choices": [{"message": {"content": "too late"}}]}, 1.5)])
        started = time.monotonic()
        try:
            run(timeout=0.4)
        except ValueError as exc:
            elapsed = time.monotonic() - started
            check("timeout raises", "超时" in str(exc), f"got {exc}")
            check("timeout does not wait for the full server delay", elapsed < 1.4, f"took {elapsed:.2f}s")
        else:
            check("timeout raises", False, "no exception raised")

        # --- retry -------------------------------------------------------------- #
        server.script(
            [
                (500, {"error": {"message": "transient"}}, 0.0),
                (503, {"error": {"message": "transient"}}, 0.0),
                (200, {"choices": [{"message": {"content": "recovered"}}]}, 0.0),
            ]
        )
        text, _ = run(max_retries=2)
        check("retry recovers from transient statuses", text == "recovered", f"got {text!r}")
        check("retry issued three attempts", len(server.requests) == 3, f"got {len(server.requests)}")

        # A configuration error must not be retried.
        server.script([(404, {"error": {"message": "no such route"}}, 0.0)])
        try:
            run(max_retries=3)
        except ValueError:
            check("404 is not retried", len(server.requests) == 1, f"got {len(server.requests)} requests")
        else:
            check("404 is not retried", False, "no exception raised")

        # --- connection failure -------------------------------------------------- #
        try:
            run(base_url="http://127.0.0.1:9/v1", timeout=2.0)
        except ValueError as exc:
            check("unreachable host raises a readable error", "LlmApiChat: " in str(exc), f"got {exc}")
        else:
            check("unreachable host raises a readable error", False, "no exception raised")
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    print("ComfyUI-LLMApi self-test")
    print(f"pack:  {PACK_DIR}")
    print(f"comfy: {COMFY_ROOT}")

    section_compile()
    try:
        pack = load_pack()
    except Exception as exc:  # noqa: BLE001
        print(f"\nFATAL: the pack failed to import: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return 1

    node_cls = section_pack(pack)
    node_module = importlib.import_module(f"{PACK_NAME}.nodes.chat")
    section_dependencies()
    section_schema(node_cls)
    section_autogrow(node_cls)
    section_locale(node_cls)
    section_packaging()
    section_url()
    section_http(node_cls, node_module)

    print(f"\n{_PASSES} passed, {len(_FAILURES)} failed")
    if _FAILURES:
        print("\nfailed checks:")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
