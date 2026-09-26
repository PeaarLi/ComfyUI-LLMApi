"""Image encoding and reasoning extraction for ``LlmApiChat``.

Two jobs, both kept out of ``chat.py`` so that file stays orchestration-only and
both are exercisable by ``_selftest.py`` without a ComfyUI server:

* turning IMAGE tensors into ``data:`` URIs an OpenAI-compatible endpoint
  accepts, and
* pulling the model's thinking out of a response whose field name varies by
  provider.

Why not reuse core's ``comfy_api_nodes.util.conversions.tensor_to_data_uri``?
That function exists, but it is core-internal: it is not re-exported from
``comfy_api_nodes.util``, so reaching it means importing a private path that a
ComfyUI update is free to move, and ``comfy_api_nodes`` is a core directory a
custom pack should not lean on. The encoding is a dozen lines against
torch/PIL, both of which every ComfyUI runtime already ships, so owning it costs
less than depending on it.
"""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

__all__ = [
    "_IMAGE_MAX_PIXELS",
    "_images_to_data_uris",
    "_reasoning_from_message",
    "_select_image_tensors",
]

# Per-image pixel budget. 1280x1280 sits under the per-image limit of every
# vision endpoint worth targeting (Claude caps near 1568x1568, OpenAI at 2048),
# while keeping the base64 payload small: these images travel inline in the JSON
# body, and base64 adds ~33% on top of PNG's already-uncompressed size, so an
# unbounded 4096x4096 input would turn one request into tens of megabytes.
_IMAGE_MAX_PIXELS = 1280 * 1280

_MIME_TYPE = "image/png"

# Ordered by how often they actually appear in the wild. DeepSeek (and the many
# relays that front it) uses ``reasoning_content``; OpenRouter normalises to
# ``reasoning``; ``reasoning_details`` is OpenRouter's structured variant.
_REASONING_STRING_KEYS = ("reasoning_content", "reasoning")
_REASONING_DETAILS_KEYS = ("reasoning_details", "reasoning_content_details")


def _select_image_tensors(images: dict | None, slot_names: list[str]) -> list[torch.Tensor]:
    """Collect the wired images in slot order, flattening any batch dimension.

    Iterating ``slot_names`` rather than the dict itself is deliberate: the
    execution engine builds the Autogrow dict, and although Python dicts keep
    insertion order, that order is the engine's business, not something this
    node should depend on. Walking the declared names makes ``image_1`` always
    the first image the model sees -- which matters when a prompt says "compare
    image 1 and image 2".

    A slot wired to a batch (``[B, H, W, C]``) contributes B separate images, so
    the caller can report an accurate count against the cap.
    """
    if not images:
        return []
    tensors: list[torch.Tensor] = []
    for name in slot_names:
        value = images.get(name)
        if value is None:
            continue
        if value.ndim == 3:
            tensors.append(value)
        elif value.ndim == 4:
            tensors.extend(value[index] for index in range(value.shape[0]))
        else:
            raise ValueError(
                f"LlmApiChat: image slot {name!r} has shape {tuple(value.shape)}; "
                "expected [H, W, C] or [B, H, W, C]."
            )
    return tensors


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a single ``[H, W, C]`` float tensor in 0..1 to a PIL image."""
    array = (image.detach().cpu().float().clamp(0.0, 1.0).contiguous().numpy() * 255.0).round().astype(np.uint8)
    if array.ndim != 3:
        raise ValueError(f"LlmApiChat: expected a 3D image tensor, got shape {tuple(array.shape)}.")
    channels = array.shape[-1]
    if channels == 1:
        # A single-channel image is a mask or a greyscale frame. Vision endpoints
        # want RGB, so replicate rather than reject.
        array = np.repeat(array, 3, axis=-1)
    elif channels not in (3, 4):
        raise ValueError(
            f"LlmApiChat: unsupported channel count {channels}; expected 1 (greyscale), 3 (RGB) or 4 (RGBA)."
        )
    return Image.fromarray(array)


def _downscale(pil_image: Image.Image) -> Image.Image:
    """Shrink to the per-image pixel budget, preserving aspect ratio.

    LANCZOS is right here (unlike the error-diffusion pipelines, which must stay
    nearest): this is a lossy downscale of photographic-ish content feeding a
    vision model, and the model reads a smoother image better than a blocky one.
    """
    width, height = pil_image.size
    if width * height <= _IMAGE_MAX_PIXELS:
        return pil_image
    scale = (_IMAGE_MAX_PIXELS / (width * height)) ** 0.5
    return pil_image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)


def _pil_to_data_uri(pil_image: Image.Image) -> str:
    buffer = BytesIO()
    # PNG keeps alpha, which a JPEG conversion would flatten onto black -- and
    # this environment's VAE produces RGBA images routinely.
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{_MIME_TYPE};base64,{encoded}"


def _images_to_data_uris(tensors: list[torch.Tensor]) -> list[str]:
    return [_pil_to_data_uri(_downscale(_tensor_to_pil(tensor))) for tensor in tensors]


def _reasoning_from_message(choice: dict) -> str:
    """Extract the model's reasoning trace, or ``""`` when the provider sent none.

    Absence is normal, not an error: only reasoning models return a trace at all,
    and every plain chat model returns nothing here. Returning an empty string
    (rather than raising or returning a placeholder) keeps a downstream node from
    having to special-case "this model does not think".

    Probed in order because the field name is not standardised -- it is a vendor
    extension that OpenAI's own spec does not define.
    """
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if not isinstance(message, dict):
        message = {}

    for key in _REASONING_STRING_KEYS:
        # Some relays hoist the trace onto the choice rather than nesting it, so
        # each key is checked in both places before moving on.
        for container in (message, choice):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value

    for key in _REASONING_DETAILS_KEYS:
        for container in (message, choice):
            details = container.get(key)
            if not isinstance(details, list):
                continue
            parts: list[str] = []
            for item in details:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                else:
                    text = item
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            if parts:
                # These are ordered fragments of one trace, not separate
                # documents, so they are concatenated without a separator.
                return "".join(parts)
    return ""
