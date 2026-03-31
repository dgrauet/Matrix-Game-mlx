"""Image transforms for video generation preprocessing.

Replaces torchvision transforms with PIL/numpy-based operations.
"""

import numpy as np
import mlx.core as mx
from PIL import Image
from typing import Callable, Tuple


def center_crop_resize(
    pil_image: Image.Image,
    target_height: int,
    target_width: int,
) -> Image.Image:
    """Resize and center-crop a PIL image to the target dimensions.

    First crops the image to match the target aspect ratio (center crop),
    then resizes to the exact target size using bicubic interpolation.

    Args:
        pil_image: Input PIL image.
        target_height: Desired output height.
        target_width: Desired output width.

    Returns:
        Center-cropped and resized PIL image.
    """
    w, h = pil_image.size
    target_ratio = target_height / target_width

    # Crop to match target aspect ratio
    if h / w > target_ratio:
        # Image is taller than target ratio — crop height
        new_h = int(w * target_ratio)
        new_w = w
    else:
        # Image is wider than target ratio — crop width
        new_h = h
        new_w = int(h / target_ratio)

    crop_y = (h - new_h) // 2
    crop_x = (w - new_w) // 2
    pil_image = pil_image.crop((crop_x, crop_y, crop_x + new_w, crop_y + new_h))

    # Resize to exact target size
    pil_image = pil_image.resize((target_width, target_height), resample=Image.BICUBIC)
    return pil_image


def get_video_transform(
    resize_height: int,
    resize_width: int,
    norm_fun: Callable[[mx.array], mx.array],
) -> Callable[[Image.Image], mx.array]:
    """Create a transform pipeline for preprocessing input images.

    The pipeline:
    1. Center-crop to match the target aspect ratio
    2. Resize to (resize_height, resize_width) via bicubic interpolation
    3. Convert to float32 in [0, 1]
    4. Apply norm_fun (typically maps [0,1] -> [-1,1])

    Args:
        resize_height: Target height (will be aligned to multiple of 32).
        resize_width: Target width (will be aligned to multiple of 32).
        norm_fun: Normalization function applied after converting to [0, 1].

    Returns:
        A callable that takes a PIL image and returns an mx.array of shape
        (H, W, C) in float32.
    """
    aligned_factor = 32
    resize_height = round(resize_height / aligned_factor) * aligned_factor
    resize_width = round(resize_width / aligned_factor) * aligned_factor

    def transform(pil_image: Image.Image) -> mx.array:
        # Ensure RGB
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Center crop and resize
        img = center_crop_resize(pil_image, resize_height, resize_width)

        # Convert to float [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)

        # Convert to mx.array and apply normalization
        x = mx.array(arr)
        x = norm_fun(x)
        return x

    return transform


if __name__ == "__main__":
    # Quick shape verification
    def normalize_to_neg_one_to_one(x: mx.array) -> mx.array:
        return 2.0 * x - 1.0

    transform = get_video_transform(720, 1280, normalize_to_neg_one_to_one)

    # Create a test image
    test_img = Image.fromarray(
        np.random.randint(0, 255, (800, 1400, 3), dtype=np.uint8)
    )
    result = transform(test_img)
    # Note: dimensions are aligned to multiples of 32
    aligned_h = round(720 / 32) * 32  # 704 (banker's rounding)
    aligned_w = round(1280 / 32) * 32  # 1280
    print(f"Transform output shape: {result.shape}")  # ({aligned_h}, {aligned_w}, 3)
    print(f"Transform output dtype: {result.dtype}")
    print(f"Value range: [{result.min().item():.2f}, {result.max().item():.2f}]")
    assert result.shape == (aligned_h, aligned_w, 3), f"Expected ({aligned_h}, {aligned_w}, 3), got {result.shape}"
    assert result.dtype == mx.float32
    print("transform.py: all checks passed")
