from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

from inference_gateway.preprocessing import (
    ImageDecodeError,
    MalformedImageError,
    UnsupportedImageFormatError,
    decode_image,
    imagenet_categories,
    preprocess_image,
    preprocess_images,
)


def _encoded_image(image_format: str, *, mode: str = "RGB") -> bytes:
    color: int | tuple[int, ...]
    if mode == "L":
        color = 80
    elif mode == "RGBA":
        color = (10, 80, 160, 120)
    else:
        color = (10, 80, 160)
    image = Image.new(mode, (41, 29), color=color)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_preprocess_uses_exact_weight_transform(image_format: str) -> None:
    payload = _encoded_image(image_format)

    actual = preprocess_image(payload)
    expected = (
        ResNet18_Weights.DEFAULT.transforms()(decode_image(payload))
        .detach()
        .cpu()
        .numpy()
    )

    assert actual.shape == (3, 224, 224)
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("mode", "image_format"),
    [("L", "PNG"), ("RGBA", "PNG"), ("RGB", "JPEG")],
)
def test_decode_converts_all_accepted_inputs_to_rgb(mode: str, image_format: str) -> None:
    decoded = decode_image(_encoded_image(image_format, mode=mode))

    assert decoded.mode == "RGB"
    assert decoded.size == (41, 29)


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\x89PNG\r\n\x1a\n"])
def test_malformed_images_raise_a_stable_error(payload: bytes) -> None:
    with pytest.raises(MalformedImageError) as error:
        decode_image(payload)

    assert isinstance(error.value, ImageDecodeError)
    assert error.value.reason == "malformed"


def test_valid_but_unsupported_image_is_distinguished() -> None:
    payload = _encoded_image("GIF")

    with pytest.raises(UnsupportedImageFormatError, match="expected JPEG or PNG") as error:
        decode_image(payload)

    assert error.value.reason == "unsupported"
    assert error.value.image_format == "GIF"


def test_decode_rejects_non_bytes_payload() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        decode_image("not bytes")  # type: ignore[arg-type]


def test_preprocess_supports_memoryview_and_injected_transform() -> None:
    payload = memoryview(_encoded_image("PNG"))

    result = preprocess_image(
        payload,
        transform=lambda image: torch.full((3, 224, 224), float(image.width)),
    )

    assert result.dtype == np.float32
    assert np.all(result == 41.0)


def test_invalid_injected_transform_outputs_are_rejected() -> None:
    payload = _encoded_image("PNG")

    with pytest.raises(TypeError, match=r"must return a torch.Tensor"):
        preprocess_image(payload, transform=lambda _image: "tensor")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"shape \[3, 224, 224\]"):
        preprocess_image(payload, transform=lambda _image: torch.zeros(3, 10, 10))


def test_preprocess_images_stacks_an_nchw_batch() -> None:
    batch = preprocess_images([_encoded_image("JPEG"), _encoded_image("PNG")])

    assert batch.shape == (2, 3, 224, 224)
    assert batch.dtype == np.float32
    assert batch.flags.c_contiguous


def test_preprocess_images_requires_at_least_one_image() -> None:
    with pytest.raises(ValueError, match="At least one image"):
        preprocess_images([])


def test_imagenet_categories_are_embedded_and_offline_safe() -> None:
    categories = imagenet_categories()

    assert len(categories) == 1000
    assert categories[0] == "tench"
    assert all(isinstance(category, str) and category for category in categories)
