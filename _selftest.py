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
import shutil
import sys
import tempfile
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

# Per-node expectations, so the schema and locale sections are table-driven
# instead of hardcoding one node's shape. A node registered without a row here
# fails the "ALL_NODES matches the expected node set" check, so the table cannot
# silently fall behind the pack.
NODE_EXPECTATIONS: dict[str, dict] = {
    "LlmApiChat": {
        "display_name": "LLM API Chat",
        "category": "text/llm api",
        "inputs": [
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
        ],
    },
    "LlmApiAnthropic": {
        "display_name": "LLM API Anthropic",
        "category": "text/llm api",
        "inputs": [
            "prompt",
            "model",
            "base_url",
            "api_key",
            "auth_style",
            "anthropic_version",
            "system_prompt",
            "skill",
            "inline_references",
            "images",
            "reasoning_trace",
            "native_skills",
            "max_tokens",
            "temperature",
            "timeout",
            "max_retries",
        ],
    },
}

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
    registered = [node.__name__ for node in nodes]
    # Set equality rather than a count: a node registered without a row in
    # NODE_EXPECTATIONS would otherwise be skipped silently by every section
    # below, which is the one way this suite could pass while covering nothing.
    check(
        "ALL_NODES matches the expected node set",
        set(registered) == set(NODE_EXPECTATIONS),
        f"got {registered}, expected {sorted(NODE_EXPECTATIONS)}",
    )

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
    return {node.__name__: node for node in nodes}


def section_schema(nodes: dict) -> None:
    section("schema")
    for node_id, expected in NODE_EXPECTATIONS.items():
        node_cls = nodes.get(node_id)
        if not check(f"{node_id}: registered", node_cls is not None):
            continue
        schema = node_cls.GET_SCHEMA()
        check(f"{node_id}: node_id", schema.node_id == node_id, f"got {schema.node_id!r}")
        check(
            f"{node_id}: display_name",
            schema.display_name == expected["display_name"],
            f"got {schema.display_name!r}",
        )
        check(f"{node_id}: category", schema.category == expected["category"], f"got {schema.category!r}")

        got_inputs = [item.id for item in schema.inputs]
        check(f"{node_id}: input ids match exactly", got_inputs == expected["inputs"], f"got {got_inputs}")

        check(f"{node_id}: exactly two outputs", len(schema.outputs) == 2, f"got {len(schema.outputs)}")
        for index, output in enumerate(schema.outputs):
            # io_type is the wire type ComfyUI routes on; the Python class of the
            # Output wrapper is the same generic class for every type.
            check(f"{node_id}: output {index} is STRING", output.io_type == "STRING", f"got {output.io_type!r}")
            check(f"{node_id}: output {index} has a display_name", bool((output.display_name or "").strip()))
            check(f"{node_id}: output {index} has a tooltip", bool((output.tooltip or "").strip()))

        blank = [item.id for item in schema.inputs if not (item.tooltip or "").strip()]
        check(f"{node_id}: every input carries a tooltip", not blank, f"blank on {blank}")

        # Required-ness is part of the API contract rather than a detail: an input
        # in `required` must be present in an API-submitted prompt, so marking
        # everything required forces a script to restate all 11/15 fields even
        # though every one of them has a default. Core's own V3 API nodes put
        # their secondary parameters in `optional`; this pins the same split.
        required = {item.id for item in schema.inputs if not getattr(item, "optional", False)}
        check(
            f"{node_id}: only prompt and model are required",
            required == {"prompt", "model"},
            f"got {sorted(required)} -- everything else has a default and should be optional",
        )

        # The Autogrow image slots must not leak into schema.inputs as separate
        # entries -- they are a single dict-typed input. If this ever changes, the
        # locale assertion below and the execute signature both need revisiting.
        check(
            f"{node_id}: images is a single input, not expanded into slots",
            "images" in got_inputs and not any(name.startswith("image_") for name in got_inputs),
            f"got {got_inputs}",
        )


def section_defaults() -> None:
    section("module constants")
    chat = importlib.import_module(f"{PACK_NAME}.nodes.chat")
    check("chat.DEFAULT_BASE_URL", chat.DEFAULT_BASE_URL == "https://api.openai.com/v1")
    check("chat temperature sentinel is negative", chat.DEFAULT_TEMPERATURE < 0)
    check("chat max_tokens sentinel omits the field", chat.DEFAULT_MAX_TOKENS == 0)
    check("chat seed sentinel omits the field", chat.DEFAULT_SEED == 0)
    check("chat MAX_IMAGES within engine limit", 0 < chat.MAX_IMAGES <= 100, f"got {chat.MAX_IMAGES}")

    anthropic = importlib.import_module(f"{PACK_NAME}.nodes.anthropic")
    check("anthropic.DEFAULT_BASE_URL", anthropic.DEFAULT_BASE_URL == "https://api.anthropic.com")
    check("anthropic temperature sentinel is negative", anthropic.DEFAULT_TEMPERATURE < 0)
    # The opposite of the sibling node's convention, and the difference is
    # deliberate: Anthropic requires max_tokens, so there is no omit sentinel and
    # the default has to be a real value.
    check(
        "anthropic max_tokens has no omit sentinel",
        anthropic.DEFAULT_MAX_TOKENS >= anthropic.MAX_TOKENS_MIN,
        f"got {anthropic.DEFAULT_MAX_TOKENS}",
    )
    check(
        "anthropic temperature range tops out at 1, not the OpenAI node's 2",
        anthropic.TEMPERATURE_MAX == 1.0,
        f"got {anthropic.TEMPERATURE_MAX}",
    )
    check("anthropic MAX_IMAGES within engine limit", 0 < anthropic.MAX_IMAGES <= 100, f"got {anthropic.MAX_IMAGES}")
    check(
        "anthropic skill options lead with the none sentinel",
        anthropic.SKILL_OPTIONS[:1] == ["(none)"],
        f"got {anthropic.SKILL_OPTIONS}",
    )
    # skills/ is not shipped (licensing), so a fresh clone discovers nothing.
    if (PACK_DIR / "skills").is_dir():
        check(
            "anthropic local skills were discovered",
            len(anthropic.SKILLS) >= 1,
            f"got {[s.name for s in anthropic.SKILLS]} - is skills/ populated?",
        )


def section_autogrow(nodes: dict) -> None:
    section("Autogrow image slots")
    from comfy_api.latest import io

    defs_source = (PACK_DIR / "locales" / "zh" / "nodeDefs.json").read_text(encoding="utf-8")
    for node_id, node_cls in sorted(nodes.items()):
        schema = node_cls.GET_SCHEMA()
        images = next((item for item in schema.inputs if item.id == "images"), None)
        if not check(f"{node_id}: images input is present", images is not None):
            continue
        if not check(
            f"{node_id}: images is an Autogrow input",
            isinstance(images, io.Autogrow.Input),
            f"got {type(images).__name__}",
        ):
            continue

        template = images.template
        check(
            f"{node_id}: template is TemplateNames",
            isinstance(template, io.Autogrow.TemplateNames),
            f"got {type(template).__name__}",
        )
        check(
            f"{node_id}: template min is 0, so no slot is required",
            getattr(template, "min", None) == 0,
            f"got {getattr(template, 'min', None)}",
        )
        names = list(getattr(template, "names", []))
        check(
            f"{node_id}: slot count matches the declared names",
            names == [f"image_{i}" for i in range(1, len(names) + 1)],
            f"got {names}",
        )
        check(f"{node_id}: slots are numbered from image_1", names[:1] == ["image_1"], f"got {names[:1]}")
        check(f"{node_id}: slot names are unique", len(names) == len(set(names)))

        # Load-bearing, and true for every node that has image slots: the frontend
        # resolves a child slot as "images_image_1" and falls back to this backend
        # string when the locale has no such key. Empty here means a blank tooltip
        # on every image slot, in every language.
        inner_tooltip = getattr(getattr(template, "input", None), "tooltip", None)
        check(
            f"{node_id}: the template's inner input carries a tooltip",
            bool((inner_tooltip or "").strip()),
            "locales/zh cannot override this one; the frontend falls back to it for every child slot",
        )

    check(
        "no image-slot locale key exists anywhere (they would be dead keys)",
        "images_image_1" not in defs_source,
        "the frontend looks up images_image_1 but the backend string wins, so such a key does nothing",
    )


def section_locale(nodes: dict) -> None:
    section("locale (zh)")
    main_path = PACK_DIR / "locales" / "zh" / "main.json"
    defs_path = PACK_DIR / "locales" / "zh" / "nodeDefs.json"
    if not check("main.json exists", main_path.is_file()) or not check("nodeDefs.json exists", defs_path.is_file()):
        return

    categories = json.loads(main_path.read_text(encoding="utf-8")).get("nodeCategories", {})
    for category in sorted({expected["category"] for expected in NODE_EXPECTATIONS.values()}):
        # The key is the LAST segment of the category path, not the whole path.
        tail = category.split("/")[-1]
        check(
            f"main.json translates category {tail!r}",
            categories.get(tail) not in (None, ""),
            f"nodeCategories={categories!r}; category was {category!r}",
        )

    specs = json.loads(defs_path.read_text(encoding="utf-8"))
    check(
        "nodeDefs.json has an entry for exactly the registered nodes",
        set(specs) == set(NODE_EXPECTATIONS),
        f"missing={sorted(set(NODE_EXPECTATIONS) - set(specs))} dead={sorted(set(specs) - set(NODE_EXPECTATIONS))}",
    )

    for node_id, node_cls in sorted(nodes.items()):
        schema = node_cls.GET_SCHEMA()
        spec = specs.get(node_id)
        if not check(f"{node_id}: nodeDefs entry exists", isinstance(spec, dict)):
            continue

        for field in ("display_name", "description"):
            check(
                f"{node_id}: {field} is non-empty",
                bool(str(spec.get(field, "")).strip()),
                f"got {spec.get(field)!r}",
            )

        # Exact equality, not "at least": a stale key left behind after a parameter
        # is renamed shows up as a blank tooltip in the UI and is very hard to spot.
        expected_inputs = {item.id for item in schema.inputs}
        got_inputs = set(spec.get("inputs", {}))
        check(
            f"{node_id}: locale input keys match the schema exactly",
            got_inputs == expected_inputs,
            f"missing={sorted(expected_inputs - got_inputs)} dead={sorted(got_inputs - expected_inputs)}",
        )

        expected_outputs = {str(index) for index in range(len(schema.outputs))}
        got_outputs = set(spec.get("outputs", {}))
        check(
            f"{node_id}: locale output keys are the positional indices",
            got_outputs == expected_outputs,
            f"missing={sorted(expected_outputs - got_outputs)} dead={sorted(got_outputs - expected_outputs)}",
        )

        for group, keys in (("inputs", expected_inputs), ("outputs", expected_outputs)):
            for key in sorted(keys & set(spec.get(group, {}))):
                entry = spec[group][key]
                check(f"{node_id}: {group}.{key} has a name", bool(str(entry.get("name", "")).strip()))
                check(f"{node_id}: {group}.{key} has a tooltip", bool(str(entry.get("tooltip", "")).strip()))


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
    # Do not join handler threads at shutdown. The timeout tests leave a handler
    # sleeping for several seconds on purpose, and joining it would make the suite
    # wait for exactly the delay the test is proving the client ignores.
    block_on_close = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[dict] = []
        self._scripted: list[tuple] = []

    def script(self, responses: list[tuple]) -> None:
        """Queue ``(status, payload, delay[, headers])``; payload may be bytes."""
        self._scripted = list(responses)
        self.requests.clear()

    def handle_error(self, request, client_address) -> None:
        """Stay quiet when a client walks away mid-request.

        The timeout tests deliberately abandon a request, and Windows reports the
        resulting teardown as ConnectionAbortedError, which socketserver would
        otherwise print as a traceback. Real stub bugs still surface.
        """
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)

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
    # It lives on the shared transport now, which both provider nodes use.
    importlib.import_module(f"{PACK_NAME}.nodes._transport")._RETRY_BASE_DELAY = 0.01

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
        # The server delay is far larger than the client timeout and the threshold
        # is correspondingly loose, on purpose. The property under test is "did not
        # wait for the response"; a threshold tight enough to be interesting would
        # instead be measuring this machine's ambient latency, which on Windows can
        # include a system proxy (httpx reads one from the registry). A loose bound
        # separates the two cleanly: waiting for the answer would mean >= 8s.
        server.script([(200, {"choices": [{"message": {"content": "too late"}}]}, 8.0)])
        started = time.monotonic()
        try:
            run(timeout=0.4)
        except ValueError as exc:
            elapsed = time.monotonic() - started
            check("timeout raises", "超时" in str(exc), f"got {exc}")
            check(
                "timeout does not wait for the full server delay",
                elapsed < 6.0,
                f"took {elapsed:.2f}s against an 8s server delay",
            )
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


def section_skills() -> None:
    section("skill loading (skills/<name>/SKILL.md)")
    skills = importlib.import_module(f"{PACK_NAME}.nodes._skills")

    root = skills._skills_root()
    bundled = skills._discover_skills()
    # The pack ships no skills/ of its own (licensing), so a fresh clone skips
    # this block; locally it runs against whatever skills/ contains.
    if root.is_dir():
        check("local skills were discovered", len(bundled) >= 1, f"got {[s.name for s in bundled]}")
        for skill in bundled:
            check(f"{skill.name}: manifest name matches its directory", skill.manifest.parent.name == skill.name)
            check(f"{skill.name}: description is non-empty", bool(skill.description.strip()))
            body = skills._load_skill_body(skill.name, inline_references=False)
            check(f"{skill.name}: body loads", bool(body.strip()))
            check(f"{skill.name}: frontmatter is stripped from the body", not body.lstrip().startswith("---"))
            check(f"{skill.name}: body keeps its heading", body.lstrip().startswith("#"))

        # A README.md next to the skill directories is not a skill, and neither
        # is anything else loose in there.
        check(
            "a loose file in skills/ is not treated as a skill",
            not any(skill.name.endswith(".md") for skill in bundled),
            f"got {[s.name for s in bundled]}",
        )
    check("the none sentinel loads nothing", skills._load_skill_body(skills._NO_SKILL) == "")
    check("an empty name loads nothing", skills._load_skill_body("") == "")
    try:
        skills._load_skill_body("no-such-skill")
    except ValueError as exc:
        message = str(exc)
        check("an unknown skill raises", "no-such-skill" in message, f"got {message}")
        if bundled:
            check("...and lists what is available", bundled[0].name in message, f"got {message}")
    else:
        check("an unknown skill raises", False, "no exception raised")

    section("skill references inlining")
    # The reason this matters: a router-style skill's body says "read
    # references/x.txt and follow it", which a single-request node cannot do. If the
    # references are not inlined the model is told to read a file it cannot reach.
    scratch = Path(tempfile.mkdtemp(prefix="llmapi-skill-refs-"))
    try:
        skill_dir = scratch / "router"
        (skill_dir / "references" / "nested").mkdir(parents=True)
        (skill_dir / "scripts").mkdir()
        (skill_dir / "assets").mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: router\ndescription: Sends you to a reference file.\n---\n\n"
            "# Router\n\nRead `references/guide.txt` and follow it.\n",
            encoding="utf-8",
        )
        (skill_dir / "references" / "guide.txt").write_text("GUIDE-CONTENT", encoding="utf-8")
        (skill_dir / "references" / "nested" / "deep.md").write_text("DEEP-CONTENT", encoding="utf-8")
        (skill_dir / "scripts" / "run.sh").write_text("SCRIPT-CONTENT", encoding="utf-8")
        (skill_dir / "assets" / "template.md").write_text("ASSET-CONTENT", encoding="utf-8")

        inlined = skills._load_skill_body("router", scratch)
        check("the body is still first", inlined.startswith("# Router"), f"got {inlined[:40]!r}")
        check("a reference file is inlined", "GUIDE-CONTENT" in inlined, f"got {inlined[-200:]!r}")
        check(
            "each inlined block is labelled with its path",
            "## Reference: references/guide.txt" in inlined,
            "the body refers to files by name, so the label is what maps instruction to content",
        )
        check("nested references are inlined too", "DEEP-CONTENT" in inlined, f"got {inlined[-200:]!r}")
        check("scripts are NOT inlined", "SCRIPT-CONTENT" not in inlined, "code to execute is not instruction text")
        check("assets are NOT inlined", "ASSET-CONTENT" not in inlined, "output templates are not instruction text")
        check(
            "references sort deterministically",
            inlined.index("GUIDE-CONTENT") < inlined.index("DEEP-CONTENT")
            or inlined.index("references/guide.txt") < inlined.index("references/nested/deep.md"),
            "an unordered walk would change the prompt between runs and break endpoint-side caching",
        )

        plain = skills._load_skill_body("router", scratch, inline_references=False)
        check("inline_references=False sends the body alone", plain == "# Router\n\nRead `references/guide.txt` and follow it.", f"got {plain!r}")

        check("_reference_files finds both reference files", len(skills._reference_files(skill_dir)) == 2)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    # The failure modes, each in its own throwaway directory. These are the cases a
    # user hits by hand-editing a SKILL.md, and the difference between "skipped
    # with a warning" and "raises" is the whole discovery/loading split.
    good = "---\nname: good\ndescription: A well-formed skill.\n---\n\n# Good\n\nBody text.\n"
    cases = {
        "good": (good, "ok"),
        "block-scalar": (
            "---\nname: block\ndescription: |\n  Multi-line description,\n  which is why this parser is PyYAML.\n---\n# B\n\nBody.\n",
            "ok",
        ),
        "no-description": ("---\nname: nodesc\n---\n\n# X\n\nBody.\n", "raise"),
        "no-frontmatter": ("# Just a heading\n\nBody.\n", "raise"),
        "unterminated": ("---\nname: u\ndescription: d\n\n# never closed\n", "raise"),
        "frontmatter-only": ("---\nname: e\ndescription: d\n---\n\n\n", "raise"),
    }
    for name, (content, expectation) in cases.items():
        scratch = Path(tempfile.mkdtemp(prefix=f"llmapi-skill-{name}-"))
        try:
            manifest = scratch / name / "SKILL.md"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(content, encoding="utf-8")

            discovered = [skill.name for skill in skills._discover_skills(scratch)]
            try:
                skills._load_skill_body(name, scratch)
            except ValueError:
                raised = True
            else:
                raised = False

            if expectation == "ok":
                check(f"{name}: discovered", discovered == [name], f"got {discovered}")
                check(f"{name}: loads without error", not raised)
            else:
                # Tolerant discovery, strict loading -- the two halves of the split.
                check(f"{name}: skipped by discovery rather than killing it", discovered == [], f"got {discovered}")
                check(f"{name}: raises when actually loaded", raised)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    # A directory with no manifest, and a loose file, are both skipped quietly.
    scratch = Path(tempfile.mkdtemp(prefix="llmapi-skill-shape-"))
    try:
        (scratch / "manifestless").mkdir()
        (scratch / "manifestless" / "notes.md").write_text("x", encoding="utf-8")
        (scratch / ".gitkeep").write_text("", encoding="utf-8")
        check(
            "a directory without SKILL.md and a loose file are both ignored",
            skills._discover_skills(scratch) == [],
            f"got {[s.name for s in skills._discover_skills(scratch)]}",
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def section_transport() -> None:
    section("proxy handling for loopback")
    transport = importlib.import_module(f"{PACK_NAME}.nodes._transport")

    # A loopback target must not be handed to a proxy. On Windows, httpx resolves
    # proxies via urllib, which falls back to the registry when HTTP_PROXY is unset,
    # and then ignores the registry's own bypass list -- so with any system proxy
    # enabled (Clash and friends) local requests go through it. Findings on this
    # machine: ProxyEnable=1, ProxyServer=127.0.0.1:7897, ProxyOverride includes
    # localhost;127.*.
    loopback = [
        "http://127.0.0.1:11434/v1",
        "http://127.0.0.1/v1",
        "http://127.5.5.5:8000/v1",  # anywhere in 127.0.0.0/8
        "http://localhost:8000/v1",
        "http://LOCALHOST:8000/v1",
        "http://localhost.localdomain/v1",
        "http://[::1]:8080/v1",
        "http://0.0.0.0:8000/v1",  # unspecified, not loopback, but means this machine
        "http://[::]:8000/v1",
    ]
    for url in loopback:
        check(f"loopback: {url}", transport._is_loopback(url), "should bypass any proxy")

    # A LAN address is not loopback: it may legitimately need a proxy, and it is
    # not this machine, so the node should keep honouring the environment.
    remote = [
        "https://api.openai.com/v1",
        "https://api.anthropic.com",
        "http://192.168.1.50:8000/v1",
        "http://10.0.0.5:8000/v1",
        "https://relay.example.com/v1",
        "",
        "not a url",
    ]
    for url in remote:
        check(f"not loopback: {url!r}", not transport._is_loopback(url), "should keep honouring ambient proxies")


def section_anthropic_url() -> None:
    section("Anthropic URL joining")
    client = importlib.import_module(f"{PACK_NAME}.nodes._anthropic_client")

    cases = [
        ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/", "https://api.anthropic.com/v1/messages"),
        # The shape the sibling node trains you to type. Appending "/v1/messages"
        # blindly would give /v1/v1/messages.
        ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1/", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1/messages", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1/messages/", "https://api.anthropic.com/v1/messages"),
        # A relay with its own prefix on the way to the same place.
        ("https://relay.example.com/anthropic", "https://relay.example.com/anthropic/v1/messages"),
        # A relay that serves the path without a version segment at all.
        ("https://relay.example.com/messages", "https://relay.example.com/messages"),
        ("  https://relay.example.com/v1  ", "https://relay.example.com/v1/messages"),
    ]
    for raw, expected in cases:
        try:
            got = client._messages_url(raw)
        except Exception as exc:  # noqa: BLE001
            check(f"messages_url({raw!r})", False, f"raised {type(exc).__name__}: {exc}")
        else:
            check(f"messages_url({raw!r}) -> {expected}", got == expected, f"got {got!r}")

    for blank in ("", "   ", None):
        try:
            client._messages_url(blank)
        except ValueError as exc:
            check(f"messages_url({blank!r}) raises", "LlmApiAnthropic: " in str(exc), f"got {exc}")
        else:
            check(f"messages_url({blank!r}) raises", False, "no exception raised")

    section("Anthropic auth headers")
    # Only one auth header may ever be sent. Asserting "the other is absent" is the
    # part that matters: a gateway rejecting an unexpected header is a real 401,
    # and asserting only presence would not catch a regression that sent both.
    x_api_key = client._auth_headers("sk-ant-abc", client._AUTH_X_API_KEY)
    check("x-api-key style sets x-api-key", x_api_key.get("x-api-key") == "sk-ant-abc", f"got {x_api_key}")
    check("x-api-key style sets no Authorization", "Authorization" not in x_api_key, f"got {x_api_key}")

    bearer = client._auth_headers("sk-ant-abc", client._AUTH_BEARER)
    check("bearer style sets Authorization", bearer.get("Authorization") == "Bearer sk-ant-abc", f"got {bearer}")
    check("bearer style sets no x-api-key", "x-api-key" not in bearer, f"got {bearer}")

    check("an empty key sends no auth header at all", client._auth_headers("") == {})
    check("a whitespace key sends no auth header at all", client._auth_headers("   ") == {})
    check("the key is stripped", client._auth_headers("  k  ") == {"x-api-key": "k"})
    check("the two styles are the advertised option list", client._AUTH_STYLES == ["x-api-key", "Authorization: Bearer"])
    try:
        client._auth_headers("k", "Basic")
    except ValueError as exc:
        check("an unknown auth_style raises", "auth_style" in str(exc), f"got {exc}")
    else:
        check("an unknown auth_style raises", False, "no exception raised")

    with_version = client._base_headers("2023-06-01")
    check("the version header is sent when set", with_version.get("anthropic-version") == "2023-06-01")
    check("Content-Type is always sent", with_version.get("Content-Type") == "application/json")
    check("a blank version omits the header", "anthropic-version" not in client._base_headers(""))
    check("a custom version is passed through", client._base_headers("2024-01-01")["anthropic-version"] == "2024-01-01")


def section_anthropic_request() -> None:
    section("Anthropic request construction")
    client = importlib.import_module(f"{PACK_NAME}.nodes._anthropic_client")

    base = {
        "max_tokens": 4096,
        "system": "",
        "temperature": -1.0,
        "reasoning_trace": False,
        "native_entries": [],
    }
    messages = [{"role": "user", "content": "hi"}]

    payload = client._build_payload("claude-opus-5", messages, **base)
    check("max_tokens is always present", payload.get("max_tokens") == 4096, f"got {payload}")
    check("stream is not sent", "stream" not in payload)
    check("a blank system omits the parameter", "system" not in payload, f"got {payload}")
    check("temperature -1 omits the field", "temperature" not in payload, f"got {payload}")
    check("reasoning_trace off omits thinking", "thinking" not in payload, f"got {payload}")
    check("no native skills means no container", "container" not in payload and "tools" not in payload, f"got {payload}")

    payload = client._build_payload(
        "claude-opus-5", messages, **{**base, "system": "be terse", "temperature": 0.0, "reasoning_trace": True}
    )
    check("system is a top-level parameter", payload.get("system") == "be terse", f"got {payload.get('system')!r}")
    check(
        "system is not smuggled into messages",
        all(message["role"] != "system" for message in payload["messages"]),
        f"got {payload['messages']}",
    )
    # temperature 0.0 is a real value and must survive; -1 is the sentinel.
    check("temperature 0.0 is sent", payload.get("temperature") == 0.0, f"got {payload.get('temperature')!r}")
    check(
        "thinking asks for a summarized trace",
        payload.get("thinking") == {"type": "adaptive", "display": "summarized"},
        f"got {payload.get('thinking')!r} - without display=summarized the trace comes back empty",
    )

    section("Anthropic system composition")
    check("skill and system prompt are joined", client._build_system("SKILL", "USER") == "SKILL\n\nUSER")
    check("the skill comes first", client._build_system("SKILL", "USER").startswith("SKILL"))
    check("a blank skill leaves the prompt alone", client._build_system("", "USER") == "USER")
    check("a blank prompt leaves the skill alone", client._build_system("SKILL", "") == "SKILL")
    check("both blank give an empty string", client._build_system("", "") == "")
    check("whitespace-only parts are dropped", client._build_system("  ", " USER ") == "USER")

    section("Anthropic user content")
    check("text-only content is a plain string", client._build_user_content("hi", []) == "hi")
    blocks = client._build_user_content("look", [("image/png", "QUJD")])
    check("an image makes the content a block list", isinstance(blocks, list), f"got {type(blocks).__name__}")
    check("the image block uses Anthropic's shape", blocks[0]["type"] == "image", f"got {blocks[0]!r}")
    check(
        "the image carries base64, not a data URI",
        blocks[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "QUJD"},
        f"got {blocks[0]['source']!r}",
    )
    check("the text block follows the image", blocks[-1] == {"type": "text", "text": "look"}, f"got {blocks[-1]!r}")
    # An image-only request is legal; an empty text block is not worth sending.
    image_only = client._build_user_content("", [("image/png", "QUJD")])
    check("a blank prompt adds no text block", len(image_only) == 1, f"got {image_only}")

    section("Anthropic native skills")
    entries = client._native_skill_entries("xlsx, skill_abc123")
    check("ids are split on commas", len(entries) == 2, f"got {entries}")
    check("a prebuilt document skill is typed anthropic", entries[0]["type"] == "anthropic", f"got {entries[0]!r}")
    check("a minted id is typed custom", entries[1]["type"] == "custom", f"got {entries[1]!r}")
    check("versions default to latest", all(entry["version"] == "latest" for entry in entries), f"got {entries}")
    check("newlines separate ids too", len(client._native_skill_entries("xlsx\npptx\npdf")) == 3)
    check("blank input yields nothing", client._native_skill_entries("") == [])
    check("whitespace-only input yields nothing", client._native_skill_entries("  \n , ") == [])
    try:
        client._native_skill_entries(",".join(f"skill_{i}" for i in range(9)))
    except ValueError as exc:
        check("more than 8 skills is rejected locally", "8" in str(exc), f"got {exc}")
    else:
        check("more than 8 skills is rejected locally", False, "no exception raised")
    check("exactly 8 skills is accepted", len(client._native_skill_entries(",".join(f"s{i}" for i in range(8)))) == 8)

    payload = client._build_payload("m", messages, **{**base, "native_entries": entries})
    check("native skills add the container", payload["container"] == {"skills": entries}, f"got {payload.get('container')}")
    check(
        "native skills add the code-execution tool",
        payload["tools"] == [{"type": "code_execution_20260521", "name": "code_execution"}],
        f"got {payload.get('tools')}",
    )


def section_anthropic_http(node_cls) -> None:
    section("Anthropic round-trip against a local stub")
    import asyncio

    client = importlib.import_module(f"{PACK_NAME}.nodes._anthropic_client")
    client._RETRY_BASE_DELAY = 0.01

    server = _StubServer(("127.0.0.1", 0), _StubHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    expected_path = "/v1/messages"

    def reply(*blocks, stop_reason="end_turn"):
        return {"content": list(blocks), "stop_reason": stop_reason}

    def text(value):
        return {"type": "text", "text": value}

    def run(**overrides):
        kwargs = {"prompt": "hello", "model": "claude-opus-5", "base_url": base_url, "api_key": "sk-ant-test"}
        kwargs.update(overrides)
        return asyncio.run(node_cls.execute(**kwargs))

    try:
        # --- happy path ------------------------------------------------------ #
        server.script([(200, reply(text("hi there")), 0.0)])
        text_out, reasoning = run()
        check("text block becomes the text output", text_out == "hi there", f"got {text_out!r}")
        check("no thinking blocks means empty reasoning", reasoning == "", f"got {reasoning!r}")

        request = server.requests[-1]
        check("posts to /v1/messages", request["path"] == expected_path, f"got {request['path']!r}")
        body = request["body"]
        check("body carries the model", body.get("model") == "claude-opus-5", f"got {body.get('model')!r}")
        check("body carries max_tokens", "max_tokens" in body, f"got {sorted(body)}")

        # --- auth styles ------------------------------------------------------ #
        check(
            "default auth sends x-api-key",
            request["headers"].get("x-api-key") == "sk-ant-test",
            f"got {request['headers'].get('x-api-key')!r}",
        )
        check(
            "...and no Authorization header",
            request["headers"].get("Authorization") is None,
            f"got {request['headers'].get('Authorization')!r}",
        )
        check(
            "the version header is sent",
            request["headers"].get("anthropic-version") == "2023-06-01",
            f"got {request['headers'].get('anthropic-version')!r}",
        )

        server.script([(200, reply(text("ok")), 0.0)])
        run(auth_style="Authorization: Bearer")
        headers = server.requests[-1]["headers"]
        check("bearer style sends Authorization", headers.get("Authorization") == "Bearer sk-ant-test", f"got {headers}")
        check("...and no x-api-key header", headers.get("x-api-key") is None, f"got {headers.get('x-api-key')!r}")

        server.script([(200, reply(text("ok")), 0.0)])
        run(api_key="")
        headers = server.requests[-1]["headers"]
        check(
            "an empty key sends neither auth header",
            headers.get("x-api-key") is None and headers.get("Authorization") is None,
            f"got {dict(headers)}",
        )

        server.script([(200, reply(text("ok")), 0.0)])
        run(anthropic_version="")
        check(
            "a blank version omits the header",
            server.requests[-1]["headers"].get("anthropic-version") is None,
            f"got {server.requests[-1]['headers'].get('anthropic-version')!r}",
        )

        # --- response shapes --------------------------------------------------- #
        server.script([(200, reply({"type": "thinking", "thinking": "step by step"}, text("answer")), 0.0)])
        text_out, reasoning = run()
        check("thinking blocks become the reasoning output", reasoning == "step by step", f"got {reasoning!r}")
        check("text still returned alongside thinking", text_out == "answer", f"got {text_out!r}")

        server.script([(200, reply(text("one "), text("two")), 0.0)])
        text_out, _ = run()
        check("multiple text blocks are concatenated", text_out == "one two", f"got {text_out!r}")

        server.script(
            [(200, reply({"type": "thinking", "thinking": "a"}, {"type": "thinking", "thinking": "b"}), 0.0)]
        )
        text_out, reasoning = run()
        check("a trace-only response leaves text empty", text_out == "", f"got {text_out!r}")
        check("...and still returns the trace", reasoning == "a\n\nb", f"got {reasoning!r}")

        server.script([(200, reply(text("cut off"), stop_reason="max_tokens"), 0.0)])
        text_out, _ = run()
        check("a truncated answer is returned, not thrown away", text_out == "cut off", f"got {text_out!r}")

        # --- error branches ---------------------------------------------------- #
        server.script(
            [(200, reply(text("no"), stop_reason="refusal") | {"stop_details": {"category": "cyber"}}, 0.0)]
        )
        try:
            run()
        except ValueError as exc:
            message = str(exc)
            check("refusal raises", "cyber" in message, f"got {message}")
            check("...and names the node", "LlmApiAnthropic: " in message, f"got {message}")
        else:
            check("refusal raises", False, "no exception raised")

        error_cases = [
            ("401", (401, {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}}, 0.0), "bad key"),
            ("400", (400, {"type": "error", "error": {"message": "model not found"}}, 0.0), "model not found"),
            ("malformed json", (200, b"<html>nope</html>", 0.0), "not valid JSON"),
            # The single most useful message in this node: an OpenAI-shaped body
            # arriving at the Anthropic node means the wrong endpoint was chosen.
            ("an OpenAI-shaped body", (200, {"choices": [{"message": {"content": "hi"}}]}, 0.0), "content"),
            ("an empty content array", (200, {"content": [], "stop_reason": "end_turn"}, 0.0), "content blocks"),
            ("empty body", (200, b"", 0.0), "empty body"),
        ]
        for name, response, needle in error_cases:
            server.script([response])
            try:
                run()
            except ValueError as exc:
                message = str(exc)
                check(f"{name} raises", "LlmApiAnthropic: " in message, f"got {message}")
                check(f"{name} message mentions the cause", needle in message, f"got {message}")
            else:
                check(f"{name} raises", False, "no exception raised")

        server.script([(401, {"type": "error", "error": {"message": "bad key"}}, 0.0)])
        try:
            run()
        except ValueError as exc:
            check("a 401 hints at the auth style", "auth_style" in str(exc), f"got {exc}")
        else:
            check("a 401 hints at the auth style", False, "no exception raised")

        # --- local skill injection --------------------------------------------- #
        # Derived from whatever skills/ contains rather than a hardcoded name, so
        # the suite survives the set changing. A fresh clone ships no skills/ at
        # all (licensing), in which case this block is skipped.
        skills_module = importlib.import_module(f"{PACK_NAME}.nodes._skills")
        anthropic_module = importlib.import_module(f"{PACK_NAME}.nodes.anthropic")
        bundled = anthropic_module.SKILLS
        if bundled:
            first_skill = bundled[0].name
            plain_body = skills_module._load_skill_body(first_skill, inline_references=False)

            server.script([(200, reply(text("ok")), 0.0)])
            run(skill=first_skill, system_prompt="USER INSTRUCTIONS")
            system = server.requests[-1]["body"].get("system", "")
            check("the skill body reaches the system parameter", system.startswith(plain_body), f"got {system[:80]!r}")
            check("...before the user's own instructions", system.index(plain_body) < system.index("USER INSTRUCTIONS"))
            check("the user's instructions are still there", system.endswith("USER INSTRUCTIONS"), f"got {system[-60:]!r}")

            # Inlining, against a real local skill that has references/.
            with_references = next((skill for skill in bundled if skills_module._reference_files(skill.directory)), None)
            if check("a local skill ships references/ to exercise inlining", with_references is not None):
                server.script([(200, reply(text("ok")), 0.0)])
                run(skill=with_references.name)
                inlined = server.requests[-1]["body"].get("system", "")
                check("references are inlined by default", "## Reference: references/" in inlined, f"got {inlined[:120]!r}")
                check(
                    "inlining adds real content",
                    len(inlined) > len(skills_module._load_skill_body(with_references.name, inline_references=False)),
                    "the system prompt should be longer with the references flattened in",
                )

                server.script([(200, reply(text("ok")), 0.0)])
                run(skill=with_references.name, inline_references=False)
                check(
                    "inline_references=False sends the body alone",
                    server.requests[-1]["body"].get("system")
                    == skills_module._load_skill_body(with_references.name, inline_references=False),
                    "got a system prompt with extra content",
                )

        server.script([(200, reply(text("ok")), 0.0)])
        run(skill="(none)", system_prompt="USER INSTRUCTIONS")
        check(
            "the none sentinel sends only the user's prompt",
            server.requests[-1]["body"].get("system") == "USER INSTRUCTIONS",
            f"got {server.requests[-1]['body'].get('system')!r}",
        )

        try:
            run(skill="no-such-skill")
        except ValueError as exc:
            check("a missing skill raises before any request", "LlmApiAnthropic: " in str(exc), f"got {exc}")
        else:
            check("a missing skill raises before any request", False, "no exception raised")

        # --- native skills ------------------------------------------------------ #
        server.script([(200, reply(text("ok")), 0.0)])
        run(native_skills="xlsx, skill_abc123")
        body = server.requests[-1]["body"]
        headers = server.requests[-1]["headers"]
        check("the container is attached", body.get("container", {}).get("skills", []) != [], f"got {body.get('container')}")
        check("the code-execution tool is attached", body.get("tools") == [{"type": "code_execution_20260521", "name": "code_execution"}], f"got {body.get('tools')}")
        betas = headers.get("anthropic-beta", "")
        check("both betas are requested", "code-execution-2025-08-25" in betas and "skills-2025-10-02" in betas, f"got {betas!r}")

        server.script([(200, reply(text("ok")), 0.0)])
        run(native_skills="")
        body = server.requests[-1]["body"]
        check(
            "no native skills means no container, no tools, no beta header",
            "container" not in body
            and "tools" not in body
            and server.requests[-1]["headers"].get("anthropic-beta") is None,
            f"container={body.get('container')!r} tools={body.get('tools')!r}",
        )

        # The container is Anthropic's own infrastructure, so a relay failing here
        # is the expected outcome -- the message has to say so rather than pass on
        # a bare 404.
        server.script([(404, {"type": "error", "error": {"message": "not found"}}, 0.0)])
        try:
            run(native_skills="xlsx")
        except ValueError as exc:
            message = str(exc)
            check("a container-path 404 explains the container prerequisite", "透传" in message, f"got {message}")
            check("...and points at the local skill route", "skill" in message, f"got {message}")
        else:
            check("a container-path 404 explains the container prerequisite", False, "no exception raised")

        # A local-skill 404 must NOT carry the container note -- it would be a lie.
        server.script([(404, {"type": "error", "error": {"message": "not found"}}, 0.0)])
        try:
            run()
        except ValueError as exc:
            check("a plain 404 carries no container note", "透传" not in str(exc), f"got {exc}")
        else:
            check("a plain 404 carries no container note", False, "no exception raised")

        # --- custom endpoint handling -------------------------------------------- #
        server.script([(200, reply(text("ok")), 0.0)])
        run(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1")
        check(
            "a base_url ending in /v1 does not double the version segment",
            server.requests[-1]["path"] == expected_path,
            f"got {server.requests[-1]['path']!r}",
        )

        server.script([(200, reply(text("ok")), 0.0)])
        run(base_url=f"http://127.0.0.1:{server.server_address[1]}/v1/messages")
        check(
            "a complete messages URL is used as-is",
            server.requests[-1]["path"] == expected_path,
            f"got {server.requests[-1]['path']!r}",
        )

        server.script([(200, reply(text("ok")), 0.0)])
        run(temperature=-1.0, reasoning_trace=False)
        body = server.requests[-1]["body"]
        check("sentinel defaults omit temperature and thinking", "temperature" not in body and "thinking" not in body, f"got {sorted(body)}")
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

    nodes = section_pack(pack)
    section_dependencies()
    section_schema(nodes)
    section_defaults()
    section_autogrow(nodes)
    section_locale(nodes)
    section_packaging()
    section_skills()
    section_transport()
    section_url()
    section_http(nodes["LlmApiChat"], importlib.import_module(f"{PACK_NAME}.nodes.chat"))
    section_anthropic_url()
    section_anthropic_request()
    section_anthropic_http(nodes["LlmApiAnthropic"])

    print(f"\n{_PASSES} passed, {len(_FAILURES)} failed")
    if _FAILURES:
        print("\nfailed checks:")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
