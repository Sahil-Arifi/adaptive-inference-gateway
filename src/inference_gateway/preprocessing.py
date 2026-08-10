"""Image decoding and model-specific preprocessing.

The preprocessing recipe comes from torchvision's weight metadata.  Accessing
that metadata and constructing the transform do not download model weights,
which keeps imports and unit tests offline-safe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import cache
from io import BytesIO
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet18_Weights

SUPPORTED_IMAGE_FORMATS = frozenset({"JPEG", "PNG"})


class ImageDecodeError(ValueError):
    """Raised when an upload cannot be decoded as an accepted image."""

    def __init__(self, message: str, *, reason: Literal["malformed", "unsupported"]) -> None:
        super().__init__(message)
        self.reason = reason


class MalformedImageError(ImageDecodeError):
    """Raised when bytes are empty, corrupt, or otherwise not an image."""

    def __init__(self, message: str = "The uploaded file is not a valid image.") -> None:
        super().__init__(message, reason="malformed")


class UnsupportedImageFormatError(ImageDecodeError):
    """Raised when a valid image uses a format other than JPEG or PNG."""

    def __init__(self, image_format: str | None) -> None:
        display_format = image_format or "unknown"
        super().__init__(
            f"Unsupported image format {display_format!r}; expected JPEG or PNG.",
            reason="unsupported",
        )
        self.image_format = image_format


def decode_image(payload: bytes | bytearray | memoryview) -> Image.Image:
    """Decode *payload*, enforce JPEG/PNG, and return an independent RGB image.

    ``Image.load`` is called while the byte stream is open so truncated or
    corrupt payloads fail here rather than later in the model transform.
    """

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("Image payload must be bytes-like.")
    raw = bytes(payload)
    if not raw:
        raise MalformedImageError("The uploaded image is empty.")

    try:
        with BytesIO(raw) as stream, Image.open(stream) as decoded:
            image_format = decoded.format.upper() if decoded.format else None
            if image_format not in SUPPORTED_IMAGE_FORMATS:
                raise UnsupportedImageFormatError(image_format)
            decoded.load()
            return decoded.convert("RGB")
    except ImageDecodeError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MalformedImageError() from exc


@cache
def _weight_transform(
    weights: ResNet18_Weights,
) -> Callable[[Image.Image], torch.Tensor]:
    return cast(Callable[[Image.Image], torch.Tensor], weights.transforms())


def preprocess_image(
    payload: bytes | bytearray | memoryview,
    *,
    weights: ResNet18_Weights = ResNet18_Weights.DEFAULT,
    transform: Callable[[Image.Image], torch.Tensor] | None = None,
) -> npt.NDArray[np.float32]:
    """Decode one upload into a contiguous ``[3, 224, 224]`` float32 tensor.

    A transform can be injected by a test, while production defaults to the
    exact preprocessing bundled with ``ResNet18_Weights.DEFAULT``.
    """

    image = decode_image(payload)
    tensor = (transform or _weight_transform(weights))(image)
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("The image transform must return a torch.Tensor.")
    if tensor.ndim != 3 or tuple(tensor.shape) != (3, 224, 224):
        raise ValueError(
            "The ResNet18 preprocessing transform must produce shape [3, 224, 224]; "
            f"received {list(tensor.shape)}."
        )
    array = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.ascontiguousarray(array, dtype=np.float32)


def preprocess_images(
    payloads: Iterable[bytes | bytearray | memoryview],
    *,
    weights: ResNet18_Weights = ResNet18_Weights.DEFAULT,
) -> npt.NDArray[np.float32]:
    """Preprocess and stack one or more images into an ``NCHW`` batch."""

    images = [preprocess_image(payload, weights=weights) for payload in payloads]
    if not images:
        raise ValueError("At least one image is required.")
    return np.ascontiguousarray(np.stack(images, axis=0), dtype=np.float32)


def imagenet_categories(
    weights: ResNet18_Weights = ResNet18_Weights.DEFAULT,
) -> tuple[str, ...]:
    """Return ImageNet category labels embedded in torchvision metadata.

    This function never downloads the actual model checkpoint.
    """

    categories = weights.meta.get("categories")
    if not isinstance(categories, list) or not all(isinstance(item, str) for item in categories):
        raise RuntimeError("The selected torchvision weights do not provide category labels.")
    return tuple(categories)


# A descriptive alias used by callers that want to emphasize the wire format.
preprocess_image_bytes = preprocess_image


__all__ = [
    "SUPPORTED_IMAGE_FORMATS",
    "ImageDecodeError",
    "MalformedImageError",
    "UnsupportedImageFormatError",
    "decode_image",
    "imagenet_categories",
    "preprocess_image",
    "preprocess_image_bytes",
    "preprocess_images",
]
