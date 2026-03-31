"""Utility functions for camera poses, Plucker ray embeddings, and data preparation.

Ported from PyTorch to MLX. Pure numpy functions (pose computation) are kept as-is.
"""

from typing import Dict, Optional, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image

from .cam_utils import (
    compute_relative_poses,
    get_extrinsics,
    get_plucker_embeddings,
    _interpolate_camera_poses_handedness,
)
from .conditions import Bench_actions_universal
from .transform import get_video_transform

WSAD_OFFSET = 12.35
DIAGONAL_OFFSET = 8.73
MOUSE_PITCH_SENSITIVITY = 15.0
MOUSE_YAW_SENSITIVITY = 15.0
MOUSE_THRESHOLD = 0.02


def compute_all_poses_from_actions(
    keyboard_conditions: np.ndarray,
    mouse_conditions: np.ndarray,
    first_pose: Optional[np.ndarray] = None,
    return_last_pose: bool = False,
) -> np.ndarray:
    """Compute all camera poses from a sequence of keyboard/mouse actions.

    Args:
        keyboard_conditions: Keyboard actions, shape (T, 4+).
        mouse_conditions: Mouse actions, shape (T, 2+).
        first_pose: Initial pose [x, y, z, pitch, yaw]. Defaults to zeros.
        return_last_pose: If True, also return the pose after the last action.

    Returns:
        All poses, shape (T, 5). If return_last_pose, returns (poses, last_pose).
    """
    T = len(keyboard_conditions)
    all_poses = np.zeros((T, 5), dtype=np.float32)
    if first_pose is not None:
        all_poses[0] = first_pose
    for i in range(T - 1):
        all_poses[i + 1] = compute_next_pose_from_action(
            all_poses[i],
            keyboard_conditions[i],
            mouse_conditions[i],
        )
    if return_last_pose:
        last_pose = compute_next_pose_from_action(
            all_poses[-1],
            keyboard_conditions[-1],
            mouse_conditions[-1],
        )
        return all_poses, last_pose
    return all_poses


def compute_next_pose_from_action(
    current_pose: np.ndarray,
    keyboard_action: np.ndarray,
    mouse_action: np.ndarray,
) -> np.ndarray:
    """Compute the next camera pose from current pose and action inputs.

    Uses average yaw for position transformation to ensure symmetric paths.
    Small mouse values below MOUSE_THRESHOLD are ignored (deadzone).

    Args:
        current_pose: Current [x, y, z, pitch, yaw].
        keyboard_action: [w, s, a, d, ...].
        mouse_action: [mouse_x, mouse_y, ...].

    Returns:
        Next pose [x, y, z, pitch, yaw].
    """
    x, y, z, pitch, yaw = current_pose
    w, s, a, d = keyboard_action[:4]
    mouse_x, mouse_y = mouse_action[:2]

    delta_pitch = MOUSE_PITCH_SENSITIVITY * mouse_x if abs(mouse_x) >= MOUSE_THRESHOLD else 0.0
    delta_yaw = MOUSE_YAW_SENSITIVITY * mouse_y if abs(mouse_y) >= MOUSE_THRESHOLD else 0.0

    new_pitch = pitch + delta_pitch
    new_yaw = yaw + delta_yaw

    while new_yaw > 180:
        new_yaw -= 360
    while new_yaw < -180:
        new_yaw += 360

    local_forward = 0.0
    if w > 0.5 and s < 0.5:
        local_forward = WSAD_OFFSET
    elif s > 0.5 and w < 0.5:
        local_forward = -WSAD_OFFSET

    local_right = 0.0
    if d > 0.5 and a < 0.5:
        local_right = WSAD_OFFSET
    elif a > 0.5 and d < 0.5:
        local_right = -WSAD_OFFSET

    if abs(local_forward) > 0.1 and abs(local_right) > 0.1:
        local_forward = np.sign(local_forward) * DIAGONAL_OFFSET
        local_right = np.sign(local_right) * DIAGONAL_OFFSET

    avg_yaw = float((yaw + new_yaw) / 2.0)
    yaw_rad = float(np.deg2rad(avg_yaw))
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    delta_x = cos_yaw * local_forward - sin_yaw * local_right
    delta_y = sin_yaw * local_forward + cos_yaw * local_right
    delta_z = 0.0

    new_x = x + delta_x
    new_y = y + delta_y
    new_z = z + delta_z

    return np.array([new_x, new_y, new_z, new_pitch, new_yaw])


def get_data(
    num_frames: int,
    height: int,
    width: int,
    pil_image: Image.Image,
) -> Tuple[mx.array, np.ndarray, mx.array, mx.array]:
    """Prepare input data for the inference pipeline.

    Transforms the input image, generates random actions, computes camera
    poses and extrinsics.

    Args:
        num_frames: Number of frames to generate.
        height: Target image height.
        width: Target image width.
        pil_image: Input PIL image.

    Returns:
        Tuple of:
        - input_image: mx.array of shape (1, 1, H, W, C) in [-1, 1]
        - extrinsics_all: numpy array of shape (T, 4, 4)
        - keyboard_condition: mx.array of shape (1, T, keyboard_dim)
        - mouse_condition: mx.array of shape (1, T, 2)
    """

    def normalize_to_neg_one_to_one(x: mx.array) -> mx.array:
        return 2.0 * x - 1.0

    transform = get_video_transform(height, width, normalize_to_neg_one_to_one)

    # Transform image: returns (H, W, C) channels-last mx.array
    input_image = transform(pil_image)  # (H, W, C)
    # Add batch and temporal dimensions: (1, 1, H, W, C)
    input_image = input_image[None, None, ...]

    actions = Bench_actions_universal(num_frames)
    keyboard_condition_all = actions["keyboard_condition"]  # (T, keyboard_dim)
    mouse_condition_all = actions["mouse_condition"]  # (T, 2)

    first_pose = np.concatenate([np.zeros(3), np.zeros(2)], axis=0)
    all_poses = compute_all_poses_from_actions(
        np.array(keyboard_condition_all),
        np.array(mouse_condition_all),
        first_pose=first_pose,
    )
    positions = all_poses[:, :3].tolist()
    rotations = np.concatenate(
        [
            np.zeros((all_poses.shape[0], 1)),  # roll = 0
            all_poses[:, 3:5],  # pitch, yaw
        ],
        axis=1,
    ).tolist()
    extrinsics_all = get_extrinsics(rotations, positions)

    return (
        input_image,
        extrinsics_all,
        keyboard_condition_all[None, ...],  # (1, T, keyboard_dim)
        mouse_condition_all[None, ...],  # (1, T, 2)
    )


def build_plucker_from_c2ws(
    c2ws_seq: mx.array,
    src_indices: np.ndarray,
    tgt_indices: np.ndarray,
    framewise: bool,
    base_K: mx.array,
    target_h: int,
    target_w: int,
    lat_h: int,
    lat_w: int,
) -> mx.array:
    """Build Plucker ray embeddings from camera-to-world matrices with interpolation.

    Args:
        c2ws_seq: Camera-to-world matrices, shape (N, 4, 4).
        src_indices: Source frame indices for interpolation.
        tgt_indices: Target frame indices.
        framewise: Whether to compute frame-to-frame relative poses.
        base_K: Base intrinsics [fx, fy, cx, cy], shape (4,).
        target_h: Target height for Plucker computation.
        target_w: Target width for Plucker computation.
        lat_h: Latent height.
        lat_w: Latent width.

    Returns:
        Plucker embeddings, shape (1, F, lat_h, lat_w, C) where C = 6 * c1 * c2.
        Channels-last layout for MLX.
    """
    c2ws_np = np.array(c2ws_seq)
    c2ws_infer_np = _interpolate_camera_poses_handedness(
        src_indices=src_indices,
        src_rot_mat=c2ws_np[:, :3, :3],
        src_trans_vec=c2ws_np[:, :3, 3],
        tgt_indices=tgt_indices,
    )
    c2ws_infer = mx.array(c2ws_infer_np.astype(np.float32))
    c2ws_infer = compute_relative_poses(c2ws_infer, framewise=framewise)

    n_frames = c2ws_infer.shape[0]
    Ks = mx.broadcast_to(base_K[None, :], (n_frames, 4))

    # plucker shape: (F, target_h, target_w, 6)
    plucker = get_plucker_embeddings(c2ws_infer, Ks, target_h, target_w)

    c1 = target_h // lat_h
    c2 = target_w // lat_w

    # Rearrange: (F, H, W, 6) -> (F, lat_h, c1, lat_w, c2, 6)
    #         -> (F, lat_h, lat_w, 6, c1, c2)
    #         -> (F, lat_h, lat_w, 6*c1*c2)
    f = len(tgt_indices)
    plucker = plucker.reshape(f, lat_h, c1, lat_w, c2, 6)
    plucker = mx.transpose(plucker, axes=(0, 1, 3, 5, 2, 4))  # (F, lat_h, lat_w, 6, c1, c2)
    plucker = plucker.reshape(f, lat_h, lat_w, 6 * c1 * c2)

    # Add batch dimension: (1, F, lat_h, lat_w, C)
    plucker = plucker[None, ...]
    return plucker


def build_plucker_from_pose(
    c2ws_pose: mx.array,
    base_K: mx.array,
    target_h: int,
    target_w: int,
    lat_h: int,
    lat_w: int,
) -> mx.array:
    """Build Plucker ray embeddings directly from camera poses.

    Args:
        c2ws_pose: Camera-to-world matrices, shape (F, 4, 4).
        base_K: Base intrinsics [fx, fy, cx, cy], shape (4,).
        target_h: Target height for Plucker computation.
        target_w: Target width for Plucker computation.
        lat_h: Latent height.
        lat_w: Latent width.

    Returns:
        Plucker embeddings, shape (1, F, lat_h, lat_w, C) where C = 6 * c1 * c2.
        Channels-last layout for MLX.
    """
    n_frames = c2ws_pose.shape[0]
    Ks = mx.broadcast_to(base_K[None, :], (n_frames, 4))

    # plucker shape: (F, target_h, target_w, 6)
    plucker = get_plucker_embeddings(c2ws_pose, Ks, target_h, target_w)

    c1 = target_h // lat_h
    c2 = target_w // lat_w

    # Rearrange: (F, H, W, 6) -> (F, lat_h, c1, lat_w, c2, 6)
    #         -> (F, lat_h, lat_w, 6, c1, c2)
    #         -> (F, lat_h, lat_w, 6*c1*c2)
    f = n_frames
    plucker = plucker.reshape(f, lat_h, c1, lat_w, c2, 6)
    plucker = mx.transpose(plucker, axes=(0, 1, 3, 5, 2, 4))  # (F, lat_h, lat_w, 6, c1, c2)
    plucker = plucker.reshape(f, lat_h, lat_w, 6 * c1 * c2)

    # Add batch dimension: (1, F, lat_h, lat_w, C)
    plucker = plucker[None, ...]
    return plucker


if __name__ == "__main__":
    # Quick shape verification
    print("=== utils.py shape verification ===")

    # Test get_data
    test_img = Image.fromarray(
        np.random.randint(0, 255, (800, 1400, 3), dtype=np.uint8)
    )
    num_frames = 57
    height, width = 720, 1280
    input_image, extrinsics_all, keyboard_cond, mouse_cond = get_data(
        num_frames, height, width, test_img
    )
    aligned_h = round(height / 32) * 32
    aligned_w = round(width / 32) * 32
    print(f"input_image shape: {input_image.shape}")  # (1, 1, {aligned_h}, {aligned_w}, 3)
    print(f"extrinsics_all shape: {extrinsics_all.shape}")  # (57, 4, 4)
    print(f"keyboard_cond shape: {keyboard_cond.shape}")  # (1, 57, 6)
    print(f"mouse_cond shape: {mouse_cond.shape}")  # (1, 57, 2)
    assert input_image.shape == (1, 1, aligned_h, aligned_w, 3), f"Expected (1, 1, {aligned_h}, {aligned_w}, 3), got {input_image.shape}"
    assert extrinsics_all.shape == (57, 4, 4)
    assert keyboard_cond.shape[0] == 1 and keyboard_cond.shape[1] == 57
    assert mouse_cond.shape == (1, 57, 2)

    # Test build_plucker_from_pose
    from .cam_utils import get_intrinsics

    base_K = get_intrinsics(height, width)
    lat_h, lat_w = 45, 80  # typical latent dimensions for 720x1280
    target_h, target_w = 720, 1280

    # Create test poses (identity-like)
    n_test_frames = 5
    c2ws = mx.broadcast_to(mx.eye(4)[None, ...], (n_test_frames, 4, 4))
    plucker = build_plucker_from_pose(
        c2ws, base_K, target_h, target_w, lat_h, lat_w
    )
    c1 = target_h // lat_h  # 16
    c2_val = target_w // lat_w  # 16
    expected_c = 6 * c1 * c2_val
    print(f"plucker shape: {plucker.shape}")  # (1, 5, 45, 80, 1536)
    assert plucker.shape == (1, n_test_frames, lat_h, lat_w, expected_c)

    print("utils.py: all checks passed")
