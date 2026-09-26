"""SKILL.md discovery and loading for the Anthropic node.

Skills on disk follow the open Agent Skills standard: a directory per skill
containing a ``SKILL.md`` whose YAML frontmatter carries ``name`` and
``description``. That is the same layout Claude Code, Codex, Gemini CLI and
Copilot CLI read -- Codex's own spec documents exactly this shape -- so the
files in this pack's ``skills/`` directory can be copied to ``~/.agents/skills/``
and used by those runtimes unchanged. Following an existing standard rather than
inventing a format is the whole point.

Two deliberate asymmetries, both documented in the node's doc page because both
surprise people:

* **Discovery happens at import, bodies are read at execute.** The Combo's
  options are built once when ComfyUI loads the pack, so adding or removing a
  skill directory needs a ComfyUI restart. Editing a skill's *body* does not --
  that is read fresh on every run.
* **Discovery is tolerant, loading is strict.** A malformed manifest is skipped
  with a log warning during discovery, because raising there would abort the
  node's ``define_schema`` and ComfyUI would swallow the exception into a single
  log line and drop the whole pack. Loading a skill the user actually selected
  raises, because that is a real error the user needs to see.

PyYAML does the frontmatter parsing. It is a direct ComfyUI dependency (it is in
ComfyUI's own requirements.txt, for extra_model_paths.yaml), so this costs no
new install. A hand-rolled parser was rejected on purpose: real SKILL.md files in
the wild use YAML block scalars for ``description`` (``description: |`` followed
by an indented block), which is exactly where a naive line-based parser breaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = [
    "_MANIFEST_NAME",
    "_NO_SKILL",
    "_Skill",
    "_discover_skills",
    "_load_skill_body",
    "_skills_root",
]

# Serves the Anthropic node; if the OpenAI node ever grows skill support too,
# this becomes a parameter rather than a constant.
_ERROR_PREFIX = "LlmApiAnthropic: "

_PACK_ROOT = Path(__file__).resolve().parent.parent
_SKILLS_DIRNAME = "skills"
_MANIFEST_NAME = "SKILL.md"

# The sentinel the combo uses for "run without a skill". Not a valid skill name
# (the spec allows only letters, digits and hyphens), so it cannot collide.
_NO_SKILL = "(none)"

# From the Agent Skills / Anthropic spec. Violations are warned about rather
# than rejected: they matter when publishing to POST /v1/skills, where the
# service validates them, but a local prompt-injected skill works fine without
# conforming, and refusing to load it would be unhelpful.
_NAME_MAX = 64
_DESCRIPTION_MAX = 1024
_REQUIRED_FIELDS = ("name", "description")


@dataclass(frozen=True)
class _Skill:
    """One discovered skill. ``body`` is deliberately absent -- see module docstring."""

    name: str
    description: str
    manifest: Path

    @property
    def directory(self) -> Path:
        return self.manifest.parent


def _skills_root() -> Path:
    return _PACK_ROOT / _SKILLS_DIRNAME


def _split_frontmatter(text: str) -> tuple[str | None, str]:
    """Split a SKILL.md into its raw frontmatter block and its body.

    Returns ``(None, text)`` when there is no frontmatter at all, and raises when
    a frontmatter block is opened with ``---`` but never closed -- that is a
    malformed file, not a file without metadata, and silently treating it as
    body-only would send the YAML to the model as prompt text.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    raise ValueError(
        f"{_MANIFEST_NAME} 以 '---' 开始但没有结束的 '---' / opens a frontmatter block that is never closed. "
        "补上第二行 '---'，或删掉第一行。/ Close the block or drop the opening delimiter."
    )


def _parse_manifest(manifest: Path) -> tuple[dict, str]:
    """Return ``(metadata, body)``, raising on anything unusable."""
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"读不了 / cannot read {manifest}: {exc}") from exc

    raw_frontmatter, body = _split_frontmatter(text)

    if raw_frontmatter is None:
        raise ValueError(
            f"{manifest} 没有 YAML frontmatter / has no YAML frontmatter. "
            f"每个 SKILL.md 都必须以 '---' 开头，并至少包含 name 和 description。"
        )

    try:
        metadata = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as exc:
        raise ValueError(f"{manifest} 的 frontmatter 不是合法 YAML / has invalid YAML frontmatter: {exc}") from exc

    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{manifest} 的 frontmatter 顶层不是映射 / frontmatter is not a YAML mapping.")

    for field in _REQUIRED_FIELDS:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{manifest} 缺少必填字段 {field!r} / is missing the required frontmatter field {field!r}. "
                f"Agent Skills 规范只强制这两个字段 / the spec requires exactly these two."
            )

    if not body.strip():
        raise ValueError(
            f"{manifest} 只有 frontmatter、没有正文 / has frontmatter but no body. "
            "frontmatter 是元数据，注入给模型的是正文。/ The body is what reaches the model."
        )
    return metadata, body


def _check_conformance(name: str, metadata: dict, manifest: Path) -> None:
    """Warn (never fail) about spec violations, which matter only on upload."""
    if len(name) > _NAME_MAX:
        logging.warning("%s skill name %r exceeds the spec's %d characters.", _ERROR_PREFIX.strip(": "), name, _NAME_MAX)
    if not all(char.isalnum() or char == "-" for char in name):
        logging.warning(
            "%s skill name %r should use only lowercase letters, digits and hyphens; "
            "POST /v1/skills will reject it otherwise.",
            _ERROR_PREFIX.strip(": "),
            name,
        )
    declared = metadata.get("name")
    if isinstance(declared, str) and declared != name:
        # Not an error: the directory name is authoritative here, because that is
        # what the combo shows and what the user picked. Worth a warning because
        # every skill examined while writing this had them identical, so a
        # mismatch usually means a mistake.
        logging.warning(
            "%s skill directory %r declares name %r in its frontmatter; using the directory name.",
            _ERROR_PREFIX.strip(": "),
            name,
            declared,
        )
    description = metadata.get("description")
    if isinstance(description, str) and len(description) > _DESCRIPTION_MAX:
        logging.warning(
            "%s skill %r has a %d-character description; the spec caps it at %d.",
            _ERROR_PREFIX.strip(": "),
            name,
            len(description),
            _DESCRIPTION_MAX,
        )


def _discover_skills(directory: Path | None = None) -> list[_Skill]:
    """List the skills under ``directory`` (default: this pack's ``skills/``).

    Tolerant by design -- see the module docstring. Returns them sorted by name so
    the combo's option order is stable across platforms (directory iteration
    order is not).
    """
    root = directory if directory is not None else _skills_root()
    if not root.is_dir():
        return []

    found: list[_Skill] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            # A stray README.md or .gitkeep in skills/ is not an error.
            continue
        manifest = child / _MANIFEST_NAME
        if not manifest.is_file():
            logging.warning(
                "%s skills/%s/ has no %s and was skipped.",
                _ERROR_PREFIX.strip(": "),
                child.name,
                _MANIFEST_NAME,
            )
            continue
        try:
            metadata, _ = _parse_manifest(manifest)
        except ValueError as exc:
            logging.warning("%s skipping skill %r: %s", _ERROR_PREFIX.strip(": "), child.name, exc)
            continue
        _check_conformance(child.name, metadata, manifest)
        found.append(_Skill(name=child.name, description=metadata["description"], manifest=manifest))
    return found


def _load_skill_body(name: str, directory: Path | None = None) -> str:
    """Read the body of the named skill. Strict, because the user picked it.

    Read fresh every call rather than cached, so editing a skill's text takes
    effect on the next run without restarting ComfyUI.
    """
    if not name or name == _NO_SKILL:
        return ""
    root = directory if directory is not None else _skills_root()
    manifest = root / name / _MANIFEST_NAME
    if not manifest.is_file():
        available = sorted(item.name for item in _discover_skills(root))
        raise ValueError(
            f"{_ERROR_PREFIX}找不到 skill {name!r} / no skill named {name!r} at {manifest}. "
            f"可用 / available: {available or '（无 / none）'}。"
            "注意 skill 目录是启动时扫描的 —— 新增目录要重启 ComfyUI。/ "
            "Skill directories are scanned at startup, so a newly added one needs a ComfyUI restart."
        )
    try:
        _, body = _parse_manifest(manifest)
    except ValueError as exc:
        raise ValueError(f"{_ERROR_PREFIX}{exc}") from exc
    # Stripped here so the caller can concatenate without thinking about the
    # blank line that always follows the closing frontmatter delimiter.
    return body.strip()
