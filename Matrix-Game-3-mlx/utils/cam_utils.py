"""Camera utility functions for Plucker ray embeddings and pose computation.

Ported from PyTorch to MLX/numpy. Functions that were pure numpy remain unchanged.
Torch tensors are replaced with mx.arrays where needed by the inference pipeline.
"""

import math
from typing import List, Tuple

import mlx.core as mx
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> np.ndarray:
    """Interpolate camera poses between source keyframes.

    Args:
        src_indices: Source frame indices.
        src_rot_mat: Source rotation matrices, shape (N, 3, 3).
        src_trans_vec: Source translation vectors, shape (N, 3).
        tgt_indices: Target frame indices to interpolate at.

    Returns:
        Interpolated 4x4 pose matrices as numpy array, shape (T, 4, 4).
    """
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()  # [N, 4]
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4), dtype=np.float32)
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return poses


def SE3_inverse(T: mx.array) -> mx.array:
    """Compute the inverse of SE(3) transformation matrices.

    Args:
        T: Batch of 4x4 transformation matrices, shape (B, 4, 4).

    Returns:
        Inverse transformation matrices, shape (B, 4, 4).
    """
    Rot = T[:, :3, :3]  # [B, 3, 3]
    trans = T[:, :3, 3:]  # [B, 3, 1]
    R_inv = mx.transpose(Rot, axes=(0, 2, 1))
    t_inv = -(R_inv @ trans)
    eye4 = mx.broadcast_to(mx.eye(4), (T.shape[0], 4, 4))
    # Build T_inv by replacing blocks
    T_inv = mx.array(eye4)
    # Use concatenation to build the result since MLX doesn't support in-place assignment
    top = mx.concatenate([R_inv, t_inv], axis=2)  # [B, 3, 4]
    bottom = mx.broadcast_to(
        mx.array([[0.0, 0.0, 0.0, 1.0]]),
        (T.shape[0], 1, 4),
    )
    T_inv = mx.concatenate([top, bottom], axis=1)  # [B, 4, 4]
    return T_inv


def compute_relative_poses(
    c2ws_mat: mx.array,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> mx.array:
    """Compute relative poses with respect to the first frame.

    Args:
        c2ws_mat: Camera-to-world matrices, shape (F, 4, 4).
        framewise: If True, compute frame-to-frame relative poses.
        normalize_trans: If True, normalize translations by max norm.

    Returns:
        Relative pose matrices, shape (F, 4, 4).
    """
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = ref_w2cs @ c2ws_mat  # broadcast matmul: (1,4,4) @ (F,4,4) -> (F,4,4)

    # Set first frame to identity
    identity = mx.eye(4, dtype=c2ws_mat.dtype)
    relative_poses = mx.concatenate(
        [identity[None, ...], relative_poses[1:]],
        axis=0,
    )

    if framewise:
        inv_prev = SE3_inverse(relative_poses[:-1])
        relative_poses_framewise = inv_prev @ relative_poses[1:]
        relative_poses = mx.concatenate(
            [relative_poses[0:1], relative_poses_framewise],
            axis=0,
        )

    if normalize_trans:
        translations = relative_poses[:, :3, 3]  # [f, 3]
        norms = mx.sqrt(mx.sum(translations * translations, axis=-1))
        max_norm = mx.max(norms)
        if max_norm.item() > 0:
            normalized_trans = translations / max_norm
            # Rebuild relative_poses with normalized translations
            rot_part = relative_poses[:, :3, :3]  # [f, 3, 3]
            top = mx.concatenate([rot_part, normalized_trans[:, :, None]], axis=2)  # [f, 3, 4]
            bottom = mx.broadcast_to(
                mx.array([[0.0, 0.0, 0.0, 1.0]]),
                (relative_poses.shape[0], 1, 4),
            )
            relative_poses = mx.concatenate([top, bottom], axis=1)

    return relative_poses


def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
) -> mx.array:
    """Create a 2D pixel coordinate grid.

    Args:
        n_frames: Number of frames to replicate the grid for.
        height: Grid height in pixels.
        width: Grid width in pixels.
        bias: Offset added to pixel coordinates (0.5 for pixel centers).

    Returns:
        Grid of (x, y) coordinates, shape (n_frames, H*W, 2).
    """
    x_range = mx.arange(width, dtype=mx.float32)
    y_range = mx.arange(height, dtype=mx.float32)
    # meshgrid: grid_y[i,j] = y_range[i], grid_x[i,j] = x_range[j]
    grid_y = mx.broadcast_to(y_range[:, None], (height, width))
    grid_x = mx.broadcast_to(x_range[None, :], (height, width))
    grid_xy = mx.stack([grid_x, grid_y], axis=-1).reshape(-1, 2) + bias  # [H*W, 2]
    grid_xy = mx.broadcast_to(grid_xy[None, ...], (n_frames, height * width, 2))
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: mx.array,
    Ks: mx.array,
    height: int,
    width: int,
) -> mx.array:
    """Compute Plucker ray embeddings from camera poses and intrinsics.

    Args:
        c2ws_mat: Camera-to-world matrices, shape (F, 4, 4).
        Ks: Intrinsic parameters [fx, fy, cx, cy], shape (F, 4).
        height: Image height.
        width: Image width.

    Returns:
        Plucker embeddings (ray_origin, ray_direction), shape (F, H, W, 6).
    """
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(n_frames, height, width)  # [f, h*w, 2]

    # Split intrinsics
    fx = Ks[:, 0:1]  # [f, 1]
    fy = Ks[:, 1:2]
    cx = Ks[:, 2:3]
    cy = Ks[:, 3:4]

    i = grid_xy[..., 0]  # [f, h*w]
    j = grid_xy[..., 1]
    zs = mx.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = mx.stack([xs, ys, zs], axis=-1)  # [f, h*w, 3]
    directions = directions / mx.sqrt(
        mx.sum(directions * directions, axis=-1, keepdims=True)
    )

    # Transform ray directions to world space
    rot = mx.transpose(c2ws_mat[:, :3, :3], axes=(0, 2, 1))  # [f, 3, 3]
    rays_d = directions @ rot  # [f, h*w, 3]

    rays_o = c2ws_mat[:, :3, 3]  # [f, 3]
    rays_o = mx.broadcast_to(rays_o[:, None, :], rays_d.shape)  # [f, h*w, 3]

    plucker_embeddings = mx.concatenate([rays_o, rays_d], axis=-1)  # [f, h*w, 6]
    plucker_embeddings = plucker_embeddings.reshape(n_frames, height, width, 6)
    return plucker_embeddings


def get_extrinsics(
    video_rotation: List[List[float]],
    video_position: List[List[float]],
) -> np.ndarray:
    """Compute extrinsic matrices from rotation angles and positions.

    Args:
        video_rotation: Per-frame [roll, pitch, yaw] in degrees, length T.
        video_position: Per-frame [x, y, z] positions, length T.

    Returns:
        Extrinsic matrices as numpy array, shape (T, 4, 4), float64.
    """
    num_frames = len(video_rotation)
    Extrinsics_vid = []
    for idx in range(num_frames):
        frame_rotation = video_rotation[idx]
        frame_position = video_position[idx]
        roll = frame_rotation[0]
        pitch = frame_rotation[1]
        yaw = frame_rotation[2]

        roll, pitch, yaw = np.radians([roll, pitch, yaw])

        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1],
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ])
        R = Rz @ Ry @ Rx
        Extrinsics = np.eye(4)
        Extrinsics[:3, :3] = R
        Extrinsics[:3, 3] = frame_position
        Extrinsics_vid.append(Extrinsics)

    R_init = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, -1, 0],
    ])
    extrinsics = np.array(Extrinsics_vid)
    extrinsics[:, :3, :3] = extrinsics[:, :3, :3] @ R_init
    extrinsics[:, :3, 3] = extrinsics[:, :3, 3] * 0.01
    return extrinsics


def get_intrinsics(height: int, width: int) -> mx.array:
    """Compute camera intrinsic parameters assuming 90-degree FOV.

    Args:
        height: Image height.
        width: Image width.

    Returns:
        Intrinsic vector [fx, fy, cx, cy] as mx.array, shape (4,).
    """
    fov_deg = 90
    fov_rad = np.deg2rad(fov_deg)

    fx = width / (2 * np.tan(fov_rad / 2))
    fy = height / (2 * np.tan(fov_rad / 2))
    cx = width / 2
    cy = height / 2

    return mx.array([fx, fy, cx, cy], dtype=mx.float32)


def _interpolate_camera_poses_handedness(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> np.ndarray:
    """Interpolate camera poses, handling left-handed coordinate systems.

    Detects left-handed rotations (negative determinant), converts to
    right-handed for SciPy interpolation, then converts back.

    Args:
        src_indices: Source frame indices.
        src_rot_mat: Source rotation matrices, shape (N, 3, 3).
        src_trans_vec: Source translation vectors, shape (N, 3).
        tgt_indices: Target frame indices.

    Returns:
        Interpolated 4x4 pose matrices as numpy array, shape (T, 4, 4).
    """
    dets = np.linalg.det(src_rot_mat)
    flip_handedness = dets.size > 0 and np.median(dets) < 0.0
    if flip_handedness:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(src_rot_mat.dtype)
        src_rot_mat = src_rot_mat @ flip_mat
    c2ws = interpolate_camera_poses(
        src_indices=src_indices,
        src_rot_mat=src_rot_mat,
        src_trans_vec=src_trans_vec,
        tgt_indices=tgt_indices,
    )
    if flip_handedness:
        c2ws[:, :3, :3] = c2ws[:, :3, :3] @ flip_mat
    return c2ws


def get_K(height: int, width: int) -> np.ndarray:
    """Compute 3x3 camera intrinsic matrix assuming 90-degree FOV.

    Args:
        height: Image height.
        width: Image width.

    Returns:
        3x3 intrinsic matrix as numpy array.
    """
    fov_deg = 90
    fov_rad = np.deg2rad(fov_deg)

    fx = width / (2 * np.tan(fov_rad / 2))
    fy = height / (2 * np.tan(fov_rad / 2))
    cx = width / 2
    cy = height / 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])
    return K


def prompt_cleaning(cap: str) -> str:
    """Clean a text prompt by removing newlines and tabs."""
    assert isinstance(cap, str)
    return cap.replace("\n", " ").replace("\t", " ")


def crop_frame(frame: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop a frame and resize back to original size."""
    w, h = frame.size
    crop_x = w // 2 - target_w // 2
    crop_y = h // 2 - target_h // 2
    cropped_frame = frame.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
    resized_frame = cropped_frame.resize((w, h), Image.BICUBIC)
    return resized_frame


def zoomed_in(
    frame: Image.Image,
    zoom_factor: float = 0.2,
    num_frames: int = 49,
) -> np.ndarray:
    """Generate a zoom-in sequence from a single frame.

    Args:
        frame: Source PIL image.
        zoom_factor: Fraction of size to zoom in.
        num_frames: Number of frames in the sequence.

    Returns:
        Array of frames, shape (num_frames, H, W, 3).
    """
    width, height = frame.size
    smallest_w = int(width * (1 - zoom_factor))
    smallest_h = int(height * (1 - zoom_factor))
    width_per_frame = np.linspace(width, smallest_w, num_frames)
    height_per_frame = np.linspace(height, smallest_h, num_frames)
    result_frames = [np.array(frame)]
    for i in range(1, num_frames):
        frame_i = crop_frame(frame, width_per_frame[i], height_per_frame[i])
        result_frames.append(np.array(frame_i))
    result_frames = np.stack(result_frames)
    return result_frames


def select_memory_idx_fov(
    extrinsics_all: np.ndarray,
    current_start_frame_idx: int,
    selected_index_base: List[int],
    return_confidence: bool = False,
) -> List[int]:
    """Select memory frame indices by FOV overlap similarity.

    For each reference frame in ``selected_index_base``, find the candidate
    frame (from indices 1..current_start_frame_idx-1) whose frustum overlaps
    the reference frustum the most. This is a numpy-only implementation
    (no GPU needed on Apple Silicon).

    Args:
        extrinsics_all: All extrinsic matrices, shape (N, 4, 4).
        current_start_frame_idx: Start of the current clip.
        selected_index_base: Reference frame indices to match against.
        return_confidence: If True, also return overlap ratios.

    Returns:
        List of selected frame indices. If ``return_confidence``, returns
        ``(indices, confidences)`` tuple.
    """
    video_w, video_h = 1280, 720
    fov_rad = np.deg2rad(90)
    fx = video_w / (2 * np.tan(fov_rad / 2))
    fy = video_h / (2 * np.tan(fov_rad / 2))

    if current_start_frame_idx <= 1:
        zeros = [0] * len(selected_index_base)
        if return_confidence:
            return zeros, [0.0] * len(selected_index_base)
        return zeros

    candidate_indices = np.arange(1, current_start_frame_idx)

    # Precompute candidate inverse transforms
    R_cand = extrinsics_all[candidate_indices, :3, :3]  # (N, 3, 3)
    t_cand = extrinsics_all[candidate_indices, :3, 3:4]  # (N, 3, 1)
    R_cand_inv = np.transpose(R_cand, (0, 2, 1))  # (N, 3, 3)
    t_cand_inv = -np.einsum("nij,njk->nik", R_cand_inv, t_cand)  # (N, 3, 1)

    # Build frustum sample points in camera space
    near, far = 0.1, 30.0
    num_side = 10
    z_samples = np.linspace(near, far, num_side)
    x_samples = np.linspace(-1, 1, num_side)
    y_samples = np.linspace(-1, 1, num_side)
    grid_x, grid_y, grid_z = np.meshgrid(x_samples, y_samples, z_samples, indexing="ij")

    points_cam_base = np.stack([
        grid_x.ravel() * grid_z.ravel() * (video_w / (2 * fx)),
        grid_y.ravel() * grid_z.ravel() * (video_h / (2 * fy)),
        grid_z.ravel(),
    ], axis=0)  # (3, M)

    selected_index: List[int] = []
    selected_confidence: List[float] = []

    for i in selected_index_base:
        E_base = extrinsics_all[i]
        points_world = E_base[:3, :3] @ points_cam_base + E_base[:3, 3:4]  # (3, M)

        # Transform world points into each candidate camera frame
        # R_cand_inv: (N, 3, 3), points_world: (3, M) -> (N, 3, M)
        points_in_cands = np.einsum(
            "nij,jk->nik", R_cand_inv, points_world
        ) + t_cand_inv  # (N, 3, M)

        x = points_in_cands[:, 0, :]
        y = points_in_cands[:, 1, :]
        z = points_in_cands[:, 2, :]

        u = (x * fx / np.maximum(z, 1e-6)) + video_w / 2
        v = (y * fy / np.maximum(z, 1e-6)) + video_h / 2

        in_view = (
            (z > near) & (z < far) &
            (u >= 0) & (u <= video_w) &
            (v >= 0) & (v <= video_h)
        )

        ratios = in_view.astype(np.float32).mean(axis=1)
        best_idx = int(np.argmax(ratios))

        selected_index.append(int(candidate_indices[best_idx]))
        selected_confidence.append(float(ratios[best_idx]))

    if return_confidence:
        return selected_index, selected_confidence
    return selected_index


def normalize_to_neg_one_to_one(x: mx.array) -> mx.array:
    """Map values from [0, 1] to [-1, 1]."""
    return 2.0 * x - 1.0
