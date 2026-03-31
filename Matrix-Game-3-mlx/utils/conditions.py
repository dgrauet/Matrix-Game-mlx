"""Predefined action sequences for benchmark testing.

Generates keyboard and mouse condition tensors for various movement
and camera actions.
"""

import random
from typing import Dict

import mlx.core as mx
import numpy as np


def combine_data(
    data: list,
    num_frames: int = 57,
    keyboard_dim: int = 4,
    mouse: bool = True,
) -> Dict[str, mx.array]:
    """Combine action data segments into full-length condition tensors.

    Args:
        data: List of dicts with 'keyboard_condition' and optionally 'mouse_condition'.
        num_frames: Total number of frames.
        keyboard_dim: Dimension of keyboard condition vector.
        mouse: Whether to include mouse conditions.

    Returns:
        Dict with 'keyboard_condition' and optionally 'mouse_condition' as mx.arrays.
    """
    assert num_frames % 4 == 1

    keyboard_np = np.zeros((num_frames, keyboard_dim), dtype=np.float32)
    if mouse:
        mouse_np = np.zeros((num_frames, 2), dtype=np.float32)

    current_frame = 0
    selections = [12]

    while current_frame < num_frames:
        rd_frame = selections[random.randint(0, len(selections) - 1)]
        rd = random.randint(0, len(data) - 1)
        k = data[rd]["keyboard_condition"]
        if mouse:
            m = data[rd]["mouse_condition"]

        if current_frame == 0:
            keyboard_np[:1] = k[:1]
            if mouse:
                mouse_np[:1] = m[:1]
            current_frame = 1
        else:
            rd_frame = min(rd_frame, num_frames - current_frame)
            repeat_time = rd_frame // 4
            # Tile the 4-frame pattern to fill rd_frame frames
            tiled_k = np.tile(k, (repeat_time, 1))
            keyboard_np[current_frame : current_frame + rd_frame] = tiled_k
            if mouse:
                tiled_m = np.tile(m, (repeat_time, 1))
                mouse_np[current_frame : current_frame + rd_frame] = tiled_m
            current_frame += rd_frame

    result = {"keyboard_condition": mx.array(keyboard_np)}
    if mouse:
        result["mouse_condition"] = mx.array(mouse_np)
    return result


def Bench_actions_universal(
    num_frames: int,
    num_samples_per_action: int = 4,
) -> Dict[str, mx.array]:
    """Generate benchmark action conditions for testing.

    Creates a diverse set of keyboard and mouse conditions covering
    single/double movement actions and camera rotations.

    Args:
        num_frames: Total number of frames.
        num_samples_per_action: Samples per action type (typically 4).

    Returns:
        Dict with 'keyboard_condition' shape (num_frames, 6) and
        'mouse_condition' shape (num_frames, 2) as mx.arrays.
    """
    actions_single_action = [
        "forward",
        "left",
        "right",
    ]
    actions_double_action = [
        "forward_left",
        "forward_right",
    ]
    actions_single_camera = [
        "camera_l",
        "camera_r",
    ]

    actions_to_test = (
        actions_double_action * 5
        + actions_single_camera * 5
        + actions_single_action * 5
    )
    for action in actions_single_action + actions_double_action:
        for camera in actions_single_camera:
            double_action = f"{action}_{camera}"
            actions_to_test.append(double_action)

    base_action = actions_single_action + actions_single_camera

    KEYBOARD_IDX = {
        "forward": 0,
        "back": 1,
        "left": 2,
        "right": 3,
    }

    CAM_VALUE = 0.1
    CAMERA_VALUE_MAP = {
        "camera_up": [CAM_VALUE, 0],
        "camera_down": [-CAM_VALUE, 0],
        "camera_l": [0, -CAM_VALUE],
        "camera_r": [0, CAM_VALUE],
        "camera_ur": [CAM_VALUE, CAM_VALUE],
        "camera_ul": [CAM_VALUE, -CAM_VALUE],
        "camera_dr": [-CAM_VALUE, CAM_VALUE],
        "camera_dl": [-CAM_VALUE, -CAM_VALUE],
    }

    data = []

    for action_name in actions_to_test:
        keyboard_condition = [
            [0, 0, 0, 0, 0, 0] for _ in range(num_samples_per_action)
        ]
        mouse_condition = [[0, 0] for _ in range(num_samples_per_action)]

        for sub_act in base_action:
            if sub_act not in action_name:
                continue
            if sub_act in CAMERA_VALUE_MAP:
                mouse_condition = [
                    CAMERA_VALUE_MAP[sub_act]
                    for _ in range(num_samples_per_action)
                ]
            elif sub_act in KEYBOARD_IDX:
                col = KEYBOARD_IDX[sub_act]
                for row in keyboard_condition:
                    row[col] = 1

        data.append({
            "keyboard_condition": np.array(keyboard_condition, dtype=np.float32),
            "mouse_condition": np.array(mouse_condition, dtype=np.float32),
        })

    return combine_data(data, num_frames, keyboard_dim=6, mouse=True)
