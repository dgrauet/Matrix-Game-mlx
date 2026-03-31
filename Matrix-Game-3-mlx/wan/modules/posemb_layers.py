"""Positional embedding layers for RoPE (Rotary Position Embedding).

MLX port of Matrix-Game-3/wan/modules/posemb_layers.py.
"""

import mlx.core as mx
from typing import Union, Tuple, List


def _to_tuple(x, dim=2):
    if isinstance(x, int):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, *args, dim=2):
    """
    Get n-D meshgrid with start, stop and num.

    Args:
        start (int or tuple): If len(args) == 0, start is num; If len(args) == 1, start is start, args[0] is stop,
            step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num. For n-dim, start/stop/num
            should be int or n-tuple. If n-tuple is provided, the meshgrid will be stacked following the dim order in
            n-tuples.
        *args: See above.
        dim (int): Dimension of the meshgrid. Defaults to 2.

    Returns:
        grid (mx.array): [dim, ...]
    """
    if len(args) == 0:
        num = _to_tuple(start, dim=dim)
        start = (0,) * dim
        stop = num
    elif len(args) == 1:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = [stop[i] - start[i] for i in range(dim)]
    elif len(args) == 2:
        start = _to_tuple(start, dim=dim)
        stop = _to_tuple(args[0], dim=dim)
        num = _to_tuple(args[1], dim=dim)
    else:
        raise ValueError(f"len(args) should be 0, 1 or 2, but got {len(args)}")

    # Build per-axis grids (matching torch.linspace with n+1 then slice [:n])
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = mx.linspace(a, b, n + 1).astype(mx.float32)[:n]
        axis_grid.append(g)

    # Meshgrid with "ij" indexing via broadcasting
    shapes = [len(g) for g in axis_grid]
    grids = []
    for i, g in enumerate(axis_grid):
        # Reshape g so it broadcasts along axis i
        shape = [1] * dim
        shape[i] = shapes[i]
        g = mx.reshape(g, shape)
        g = mx.broadcast_to(g, shapes)
        grids.append(g)

    grid = mx.stack(grids, axis=0)  # [dim, ...]
    return grid


#################################################################################
#                   Rotary Positional Embedding Functions                       #
#################################################################################
# https://github.com/meta-llama/llama/blob/be327c427cc5e89cc1d3ab3d3fec4484df771245/llama/model.py#L80


def reshape_for_broadcast(
    freqs_cis: Tuple[mx.array, mx.array],
    x: mx.array,
    head_first: bool = False,
) -> Tuple[mx.array, mx.array]:
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Tuple[mx.array, mx.array]): Frequency tensor (cos, sin) to be reshaped.
        x (mx.array): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[mx.array, mx.array]: Reshaped frequency tensors (cos, sin).

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    # freqs_cis: (cos, sin) in real space
    if head_first:
        assert freqs_cis[0].shape == (
            x.shape[-2],
            x.shape[-1],
        ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
        shape = [
            d if i == ndim - 2 or i == ndim - 1 else 1
            for i, d in enumerate(x.shape)
        ]
    else:
        assert freqs_cis[0].shape == (
            x.shape[1],
            x.shape[-1],
        ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return mx.reshape(freqs_cis[0], shape), mx.reshape(freqs_cis[1], shape)


def rotate_half(x: mx.array) -> mx.array:
    """Rotate half of the hidden dims of the input."""
    x = x.astype(mx.float32)
    # Reshape last dim into pairs: [..., D] -> [..., D//2, 2]
    x = mx.reshape(x, (*x.shape[:-1], -1, 2))
    x_real = x[..., 0]  # [..., D//2]
    x_imag = x[..., 1]  # [..., D//2]
    # Stack [-x_imag, x_real] and flatten back
    out = mx.stack([-x_imag, x_real], axis=-1)  # [..., D//2, 2]
    return mx.reshape(out, (*out.shape[:-2], -1))  # [..., D]


def apply_rotary_emb(
    xq: mx.array,
    xk: mx.array,
    freqs_cis: Tuple[mx.array, mx.array],
    head_first: bool = False,
) -> Tuple[mx.array, mx.array]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (mx.array): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (mx.array): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (Tuple[mx.array, mx.array]): Precomputed frequency tensor (cos, sin).
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[mx.array, mx.array]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    assert isinstance(freqs_cis, tuple)
    cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
    xq_out = (xq.astype(mx.float32) * cos[:, :xq.shape[1], :, :] +
              rotate_half(xq.astype(mx.float32)) * sin[:, :xq.shape[1], :, :]).astype(xq.dtype)
    xk_out = (xk.astype(mx.float32) * cos[:, :xk.shape[1], :, :] +
              rotate_half(xk.astype(mx.float32)) * sin[:, :xk.shape[1], :, :]).astype(xk.dtype)
    return xq_out, xk_out


def get_nd_rotary_pos_embed(
    rope_dim_list,
    start,
    *args,
    theta=10000.0,
    use_real=False,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        start (int | tuple of int | list of int): If len(args) == 0, start is num; If len(args) == 1, start is start,
            args[0] is stop, step is 1; If len(args) == 2, start is start, args[0] is stop, args[1] is num.
        *args: See above.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool): If True, return real part and imaginary part separately. Otherwise, return complex numbers.
            Some libraries such as TensorRT does not support complex64 data type. So it is useful to provide a real
            part and an imaginary part separately.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (mx.array): [HW, D/2]
    """

    grid = get_meshgrid_nd(
        start, *args, dim=len(rope_dim_list)
    )

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)
    assert len(theta_rescale_factor) == len(
        rope_dim_list
    ), "len(theta_rescale_factor) should equal to len(rope_dim_list)"

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)
    assert len(interpolation_factor) == len(
        rope_dim_list
    ), "len(interpolation_factor) should equal to len(rope_dim_list)"

    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(
            rope_dim_list[i],
            mx.reshape(grid[i], (-1,)),
            theta,
            use_real=use_real,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )
        embs.append(emb)

    if use_real:
        cos = mx.concatenate([emb[0] for emb in embs], axis=1)
        sin = mx.concatenate([emb[1] for emb in embs], axis=1)
        return cos, sin
    else:
        emb = mx.concatenate(embs, axis=1)
        return emb


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[mx.array, int],
    theta: float = 10000.0,
    use_real: bool = False,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
) -> Union[mx.array, Tuple[mx.array, mx.array]]:
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.
    (Note: `cis` means `cos + i * sin`, where i is the imaginary unit.)

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or mx.array): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (bool, optional): If True, return real part and imaginary part separately.
                                   Otherwise, return complex numbers.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

    Returns:
        freqs_cis: Precomputed frequency tensor with complex exponential. [S, D/2]
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = mx.arange(pos).astype(mx.float32)

    # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
    # has some connection to NTK literature
    if theta_rescale_factor != 1.0:
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta ** (mx.arange(0, dim, 2)[: (dim // 2)].astype(mx.float32) / dim)
    )
    # outer product: [S] x [D/2] -> [S, D/2]
    freqs = mx.expand_dims(pos * interpolation_factor, 1) * mx.expand_dims(freqs, 0)

    if use_real:
        freqs_cos = mx.cos(freqs)  # [S, D/2]
        freqs_sin = mx.sin(freqs)  # [S, D/2]
        # repeat_interleave(2, dim=1): [S, D/2] -> [S, D]
        freqs_cos = mx.reshape(
            mx.broadcast_to(mx.expand_dims(freqs_cos, -1), (*freqs_cos.shape, 2)),
            (freqs_cos.shape[0], -1),
        )
        freqs_sin = mx.reshape(
            mx.broadcast_to(mx.expand_dims(freqs_sin, -1), (*freqs_sin.shape, 2)),
            (freqs_sin.shape[0], -1),
        )
        return freqs_cos, freqs_sin
    else:
        # Return (cos, sin) tuple since MLX doesn't have complex numbers
        return mx.cos(freqs), mx.sin(freqs)
