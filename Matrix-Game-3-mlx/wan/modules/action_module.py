"""ActionModule: mouse/keyboard conditioning for the DiT backbone.

MLX port of Matrix-Game-3/wan/modules/action_module.py.
"""

import math
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .model import WanRMSNorm
from .posemb_layers import apply_rotary_emb, get_nd_rotary_pos_embed

__all__ = ["ActionModule"]


class ActionModule(nn.Module):
    """Action module from https://arxiv.org/pdf/2501.08325.

    Conditions video generation on mouse movements and keyboard inputs
    via cross-attention into the DiT hidden states.

    Args:
        mouse_dim_in: Input dimension for mouse condition.
        keyboard_dim_in: Input dimension for keyboard condition.
        hidden_size: Hidden dimension for keyboard embedding.
        img_hidden_size: Hidden dimension of the image/DiT features.
        keyboard_hidden_dim: Hidden dim for keyboard attention path.
        mouse_hidden_dim: Hidden dim for mouse attention path.
        vae_time_compression_ratio: Temporal compression factor of the VAE.
        windows_size: Number of temporal windows for conditioning.
        heads_num: Number of attention heads.
        patch_size: Patch dimensions [t, h, w].
        qk_norm: Whether to apply QK normalization.
        qkv_bias: Whether to use bias in QKV projections.
        rope_dim_list: RoPE dimension splits [t, h, w].
        rope_theta: RoPE base frequency.
        mouse_qk_dim_list: RoPE dimension splits for mouse QK.
        enable_mouse: Whether to enable mouse conditioning.
        enable_keyboard: Whether to enable keyboard conditioning.
        blocks: Unused, kept for API compatibility.
        local_attn_size: Local attention window size.
    """

    def __init__(
        self,
        mouse_dim_in: int = 2,
        keyboard_dim_in: int = 6,
        hidden_size: int = 128,
        img_hidden_size: int = 1536,
        keyboard_hidden_dim: int = 1024,
        mouse_hidden_dim: int = 1024,
        vae_time_compression_ratio: int = 4,
        windows_size: int = 3,
        heads_num: int = 16,
        patch_size: list = [1, 2, 2],
        qk_norm: bool = True,
        qkv_bias: bool = False,
        rope_dim_list: list = [8, 28, 28],
        rope_theta: float = 256,
        mouse_qk_dim_list: list = [8, 28, 28],
        enable_mouse: bool = True,
        enable_keyboard: bool = True,
        blocks: list = [],
        local_attn_size: int = 6,
    ):
        super().__init__()
        self.local_attn_size = local_attn_size
        self.enable_mouse = enable_mouse
        self.enable_keyboard = enable_keyboard
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.mouse_qk_dim_list = mouse_qk_dim_list
        self.heads_num = heads_num
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size
        self.patch_size = patch_size

        if self.enable_keyboard:
            # nn.Sequential replacement: keyboard_embed is [Linear, SiLU, Linear]
            self.keyboard_embed_linear1 = nn.Linear(keyboard_dim_in, hidden_size, bias=True)
            self.keyboard_embed_linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

        if self.enable_mouse:
            c = mouse_hidden_dim
            mouse_mlp_in = mouse_dim_in * vae_time_compression_ratio * windows_size + img_hidden_size
            self.mouse_mlp_linear1 = nn.Linear(mouse_mlp_in, c, bias=True)
            self.mouse_mlp_linear2 = nn.Linear(c, c, bias=False)
            self.mouse_mlp_layernorm = nn.LayerNorm(c)

            head_dim = c // heads_num
            self.t_qkv = nn.Linear(c, c * 3, bias=qkv_bias)
            self.img_attn_q_norm = WanRMSNorm(head_dim, eps=1e-6) if qk_norm else None
            self.img_attn_k_norm = WanRMSNorm(head_dim, eps=1e-6) if qk_norm else None
            self.proj_mouse = nn.Linear(c, img_hidden_size, bias=qkv_bias)

        if self.enable_keyboard:
            head_dim_key = keyboard_hidden_dim // heads_num
            self.key_attn_q_norm = WanRMSNorm(head_dim_key, eps=1e-6) if qk_norm else None
            self.key_attn_k_norm = WanRMSNorm(head_dim_key, eps=1e-6) if qk_norm else None
            self.mouse_attn_q = nn.Linear(img_hidden_size, keyboard_hidden_dim, bias=qkv_bias)
            self.keyboard_attn_kv = nn.Linear(
                hidden_size * windows_size * vae_time_compression_ratio,
                keyboard_hidden_dim * 2,
                bias=qkv_bias,
            )
            self.proj_keyboard = nn.Linear(keyboard_hidden_dim, img_hidden_size, bias=qkv_bias)

    def _keyboard_embed(self, x: mx.array) -> mx.array:
        """Apply keyboard embedding: Linear -> SiLU -> Linear."""
        x = self.keyboard_embed_linear1(x)
        x = nn.silu(x)
        x = self.keyboard_embed_linear2(x)
        return x

    def _mouse_mlp(self, x: mx.array) -> mx.array:
        """Apply mouse MLP: Linear -> GELU(tanh) -> Linear -> LayerNorm."""
        x = self.mouse_mlp_linear1(x)
        x = nn.gelu_approx(x)
        x = self.mouse_mlp_linear2(x)
        x = self.mouse_mlp_layernorm(x)
        return x

    def patchify(self, x: mx.array, patch_size: int) -> mx.array:
        """Patchify a 5D tensor.

        Args:
            x: Input tensor [N, C, T, H, W].
            patch_size: Total patch volume (pt * ph * pw).

        Returns:
            Patched tensor [N, T*H*W, C*pt*ph*pw].
        """
        pt, ph, pw = self.patch_size
        N, c = x.shape[0], x.shape[1]
        t = x.shape[2] // pt
        h = x.shape[3] // ph
        w = x.shape[4] // pw
        # reshape: [N, c, t, pt, h, ph, w, pw]
        x = x.reshape(N, c, t, pt, h, ph, w, pw)
        # einsum "nctohpwq->nthwcopq" is a transpose:
        # (0,1,2,3,4,5,6,7) -> (0,2,4,6,1,3,5,7)
        x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        # reshape: [N, t*h*w, c*pt*ph*pw]
        x = x.reshape(N, t * h * w, c * pt * ph * pw)
        return x

    def unpatchify(
        self, x: mx.array, t: int, h: int, w: int, patch_size: int
    ) -> mx.array:
        """Unpatchify back to 5D tensor.

        Args:
            x: Patched tensor [N, T, patch_size * C].
            t: Number of temporal patches.
            h: Number of height patches.
            w: Number of width patches.
            patch_size: Total patch volume.

        Returns:
            Unpatched tensor [N, C, T*pt, H*ph, W*pw].
        """
        c = x.shape[2] // patch_size
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        # reshape: [N, t, h, w, c, pt, ph, pw]
        x = x.reshape(x.shape[0], t, h, w, c, pt, ph, pw)
        # einsum "nthwcopq->nctohpwq" is a transpose:
        # (0,1,2,3,4,5,6,7) -> (0,4,1,5,2,6,3,7)
        x = x.transpose(0, 4, 1, 5, 2, 6, 3, 7)
        # reshape: [N, c, t*pt, h*ph, w*pw]
        x = x.reshape(x.shape[0], c, t * pt, h * ph, w * pw)
        return x

    def get_rotary_pos_embed(
        self,
        video_length: int,
        height: int,
        width: int,
        head_dim: int,
        rope_dim_list: Optional[List[int]] = None,
    ) -> tuple:
        """Compute rotary positional embeddings for the action module.

        Args:
            video_length: Number of temporal frames.
            height: Spatial height.
            width: Spatial width.
            head_dim: Head dimension.
            rope_dim_list: Optional RoPE dimension splits.

        Returns:
            Tuple of (freqs_cos, freqs_sin).
        """
        target_ndim = 3
        latents_size = [video_length, height, width]

        if isinstance(self.patch_size, int):
            rope_sizes = [s // self.patch_size for s in latents_size]
        elif isinstance(self.patch_size, list):
            rope_sizes = [
                s // self.patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes

        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, (
            "sum(rope_dim_list) should equal to head_dim of attention layer"
        )

        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        cutoff = video_length * rope_sizes[1] * rope_sizes[2] // self.patch_size[0]
        return freqs_cos[-cutoff:], freqs_sin[-cutoff:]

    def __call__(
        self,
        x: mx.array,
        tt: int,
        th: int,
        tw: int,
        mouse_condition: Optional[mx.array] = None,
        keyboard_condition: Optional[mx.array] = None,
        mouse_cond_memory: Optional[mx.array] = None,
        keyboard_cond_memory: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass of the action module.

        Args:
            x: Hidden states [B, tt*th*tw, C].
            tt: Number of temporal tokens.
            th: Number of height tokens.
            tw: Number of width tokens.
            mouse_condition: Mouse input [B, N_frames, C1].
            keyboard_condition: Keyboard input [B, N_frames, C2].
            mouse_cond_memory: Optional mouse memory conditioning.
            keyboard_cond_memory: Optional keyboard memory conditioning.

        Returns:
            Updated hidden states [B, tt*th*tw, C].
        """
        B_orig = x.shape[0]
        N_frames = keyboard_condition.shape[1]
        assert tt * th * tw == x.shape[1]

        cond1 = ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0
        cond2 = N_frames % self.vae_time_compression_ratio == 0
        assert cond1 or cond2

        if cond1:
            N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1
        else:
            N_feats = N_frames // self.vae_time_compression_ratio

        memory_length = 0
        pad_t = self.vae_time_compression_ratio * self.windows_size

        # =====================================================================
        # Mouse conditioning path
        # =====================================================================
        if self.enable_mouse and mouse_condition is not None:
            B = B_orig
            S = th * tw
            # rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=th*tw)
            # x: [B, tt*S, C] -> [B, tt, S, C] -> [B, S, tt, C] -> [B*S, tt, C]
            hidden_states = x.reshape(B, tt, S, -1).transpose(0, 2, 1, 3).reshape(B * S, tt, -1)

            N_frames_m = mouse_condition.shape[1]
            C_m = mouse_condition.shape[2]

            # Pad mouse condition
            if cond1:
                pad = mx.broadcast_to(mouse_condition[:, 0:1, :], (B, pad_t, C_m))
                mouse_condition = mx.concatenate([pad, mouse_condition], axis=1)
            else:
                pad = mx.broadcast_to(mouse_condition[:, 0:1, :], (B, pad_t - 4, C_m))
                mouse_condition = mx.concatenate([pad, mouse_condition], axis=1)

            # Group mouse conditions per temporal block
            group_mouse_list = []
            for i in range(N_feats):
                start_idx = self.vae_time_compression_ratio * (i - self.windows_size) + pad_t
                end_idx = i * self.vae_time_compression_ratio + pad_t
                group_mouse_list.append(mouse_condition[:, start_idx:end_idx, :])
            group_mouse = mx.stack(group_mouse_list, axis=1)  # [B, N_feats, window, D]

            if mouse_cond_memory is not None:
                memory_length = mouse_cond_memory.shape[1]
                # [B, mem, D] -> [B, mem, 1, D] -> [B, mem, pad_t, D]
                mouse_cond_memory = mx.broadcast_to(
                    mx.expand_dims(mouse_cond_memory, 2),
                    (B, memory_length, pad_t, mouse_cond_memory.shape[-1]),
                )
                group_mouse = mx.concatenate([mouse_cond_memory, group_mouse], axis=1)

            # [B, T, window, D] -> [B, T, window, D, 1] -> [B, T, window, D, S]
            group_mouse = mx.broadcast_to(
                mx.expand_dims(group_mouse, -1),
                (*group_mouse.shape, S),
            )

            # rearrange('b t window d s -> (b s) t (window d)')
            # [B, T, window, D, S] -> [B, S, T, window, D] -> [B*S, T, window*D]
            group_mouse = group_mouse.transpose(0, 4, 1, 2, 3)
            group_mouse = group_mouse.reshape(B * S, group_mouse.shape[2], -1)

            # Concatenate with hidden states and pass through MLP
            group_mouse = mx.concatenate([hidden_states, group_mouse], axis=-1)
            group_mouse = self._mouse_mlp(group_mouse)

            # QKV projection
            mouse_qkv = self.t_qkv(group_mouse)
            # rearrange("B L (K H D) -> K B L H D", K=3, H=heads_num)
            BxS, L, _ = mouse_qkv.shape
            H = self.heads_num
            D = mouse_qkv.shape[-1] // (3 * H)
            mouse_qkv = mouse_qkv.reshape(BxS, L, 3, H, D)
            mouse_qkv = mouse_qkv.transpose(2, 0, 1, 3, 4)  # [3, BxS, L, H, D]
            q, k, v = mouse_qkv[0], mouse_qkv[1], mouse_qkv[2]

            # QK normalization
            if self.img_attn_q_norm is not None:
                q = self.img_attn_q_norm(q)
            if self.img_attn_k_norm is not None:
                k = self.img_attn_k_norm(k)

            # Apply RoPE
            if memory_length > 0:
                freqs_cos_mem, freqs_sin_mem = self.get_rotary_pos_embed(
                    memory_length, self.patch_size[1], self.patch_size[2],
                    k.shape[-1], self.mouse_qk_dim_list,
                )
                qq_mem, kk_mem = apply_rotary_emb(
                    q[:, :memory_length], k[:, :memory_length],
                    (freqs_cos_mem, freqs_sin_mem), head_first=False,
                )
                freqs_cos_pred, freqs_sin_pred = self.get_rotary_pos_embed(
                    tt - memory_length, self.patch_size[1], self.patch_size[2],
                    k.shape[-1], self.mouse_qk_dim_list,
                )
                qq_pred, kk_pred = apply_rotary_emb(
                    q[:, memory_length:], k[:, memory_length:],
                    (freqs_cos_pred, freqs_sin_pred), head_first=False,
                )
                q = mx.concatenate([qq_mem, qq_pred], axis=1)
                k = mx.concatenate([kk_mem, kk_pred], axis=1)
            else:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                    tt, self.patch_size[1], self.patch_size[2],
                    k.shape[-1], self.mouse_qk_dim_list,
                )
                qq, kk = apply_rotary_emb(q, k, (freqs_cos, freqs_sin), head_first=False)
                q, k = qq, kk

            # Attention: q,k,v are [BxS, L, H, D], need [BxS, H, L, D] for SDPA
            scale = 1.0 / math.sqrt(q.shape[-1])
            q_t = q.transpose(0, 2, 1, 3)
            k_t = k.transpose(0, 2, 1, 3)
            v_t = v.transpose(0, 2, 1, 3)
            attn = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
            attn = attn.transpose(0, 2, 1, 3)  # back to [BxS, L, H, D]

            # rearrange(attn, '(b S) T h d -> b (T S) (h d)', b=B)
            # attn: [B*S, T, H, D] -> [B, S, T, H, D] -> [B, T, S, H, D] -> [B, T*S, H*D]
            attn = attn.reshape(B, S, attn.shape[1], H, D)
            attn = attn.transpose(0, 2, 1, 3, 4)
            attn = attn.reshape(B, attn.shape[1] * S, H * D)

            # Reconstruct hidden_states from the original rearranged x
            # rearrange(x, "(B S) T C -> B (T S) C", B=B)
            # hidden_states was [B*S, T, C], need [B, T*S, C]
            # Recompute from x which was [B*S, tt, C_img]
            C_img = x.shape[-1]
            hidden_states = hidden_states.reshape(B, S, tt, C_img)
            hidden_states = hidden_states.transpose(0, 2, 1, 3)
            hidden_states = hidden_states.reshape(B, tt * S, C_img)

            attn = self.proj_mouse(attn)
            hidden_states = hidden_states + attn
        else:
            hidden_states = x

        # =====================================================================
        # Keyboard conditioning path
        # =====================================================================
        if self.enable_keyboard and keyboard_condition is not None:
            N_frames_k = keyboard_condition.shape[1]
            C_k = keyboard_condition.shape[2]

            # Pad keyboard condition
            if cond1:
                pad = mx.broadcast_to(keyboard_condition[:, 0:1, :], (B_orig, pad_t, C_k))
                keyboard_condition = mx.concatenate([pad, keyboard_condition], axis=1)
            else:
                pad = mx.broadcast_to(keyboard_condition[:, 0:1, :], (B_orig, pad_t - 4, C_k))
                keyboard_condition = mx.concatenate([pad, keyboard_condition], axis=1)

            keyboard_condition = self._keyboard_embed(keyboard_condition)

            group_keyboard_list = []
            for i in range(N_feats):
                start_idx = self.vae_time_compression_ratio * (i - self.windows_size) + pad_t
                end_idx = i * self.vae_time_compression_ratio + pad_t
                group_keyboard_list.append(keyboard_condition[:, start_idx:end_idx, :])
            group_keyboard = mx.stack(group_keyboard_list, axis=1)  # [B, N_feats, window, D]

            if keyboard_cond_memory is not None:
                memory_length = keyboard_cond_memory.shape[1]
                keyboard_cond_memory_emb = self._keyboard_embed(keyboard_cond_memory)
                # [B, mem, D] -> [B, mem, 1, D] -> [B, mem, pad_t, D]
                keyboard_cond_memory_emb = mx.broadcast_to(
                    mx.expand_dims(keyboard_cond_memory_emb, 2),
                    (B_orig, memory_length, pad_t, keyboard_cond_memory_emb.shape[-1]),
                )
                group_keyboard = mx.concatenate([keyboard_cond_memory_emb, group_keyboard], axis=1)

            # Flatten window dim: [B, T, window, D] -> [B, T, window*D]
            group_keyboard = group_keyboard.reshape(
                group_keyboard.shape[0], group_keyboard.shape[1], -1
            )

            # Query from hidden states, KV from keyboard
            mouse_q = self.mouse_attn_q(hidden_states)
            keyboard_kv = self.keyboard_attn_kv(group_keyboard)

            # rearrange(mouse_q, "B L (H D) -> B L H D", H=heads_num)
            H = self.heads_num
            D_key = mouse_q.shape[-1] // H
            q = mouse_q.reshape(mouse_q.shape[0], mouse_q.shape[1], H, D_key)

            # rearrange(keyboard_kv, "B L (K H D) -> K B L H D", K=2, H=heads_num)
            D_kv = keyboard_kv.shape[-1] // (2 * H)
            keyboard_kv = keyboard_kv.reshape(
                keyboard_kv.shape[0], keyboard_kv.shape[1], 2, H, D_kv
            )
            keyboard_kv = keyboard_kv.transpose(2, 0, 1, 3, 4)  # [2, B, L, H, D]
            k, v = keyboard_kv[0], keyboard_kv[1]

            # QK normalization
            if self.key_attn_q_norm is not None:
                q = self.key_attn_q_norm(q)
            if self.key_attn_k_norm is not None:
                k = self.key_attn_k_norm(k)

            S = th * tw
            # rearrange(q, "B (T S) H D -> (B S) T H D", S=S)
            # q: [B, T*S, H, D] -> [B, T, S, H, D] -> [B, S, T, H, D] -> [B*S, T, H, D]
            T_q = q.shape[1] // S
            q = q.reshape(B_orig, T_q, S, H, D_key)
            q = q.transpose(0, 2, 1, 3, 4)
            q = q.reshape(B_orig * S, T_q, H, D_key)

            # Apply RoPE
            if memory_length > 0:
                freqs_cos_mem, freqs_sin_mem = self.get_rotary_pos_embed(
                    memory_length, self.patch_size[1], self.patch_size[2],
                    k.shape[-1], self.mouse_qk_dim_list,
                )
                qq_mem, kk_mem = apply_rotary_emb(
                    q[:, :memory_length], k[:, :memory_length],
                    (freqs_cos_mem, freqs_sin_mem), head_first=False,
                )
                freqs_cos_pred, freqs_sin_pred = self.get_rotary_pos_embed(
                    tt - memory_length, self.patch_size[1], self.patch_size[2],
                    k.shape[-1], self.mouse_qk_dim_list,
                )
                qq_pred, kk_pred = apply_rotary_emb(
                    q[:, memory_length:], k[:, memory_length:],
                    (freqs_cos_pred, freqs_sin_pred), head_first=False,
                )
                q = mx.concatenate([qq_mem, qq_pred], axis=1)
                k = mx.concatenate([kk_mem, kk_pred], axis=1)
            else:
                freqs_cos, freqs_sin = self.get_rotary_pos_embed(
                    tt, self.patch_size[1], self.patch_size[2],
                    k.shape[-1], self.rope_dim_list,
                )
                qq, kk = apply_rotary_emb(q, k, (freqs_cos, freqs_sin), head_first=False)
                q, k = qq, kk

            # Repeat k, v for each spatial position: [B, L, H, D] -> [B*S, L, H, D]
            k = mx.concatenate([k] * S, axis=0)
            v = mx.concatenate([v] * S, axis=0)

            # Attention
            scale = 1.0 / math.sqrt(q.shape[-1])
            q_t = q.transpose(0, 2, 1, 3)
            k_t = k.transpose(0, 2, 1, 3)
            v_t = v.transpose(0, 2, 1, 3)
            attn = mx.fast.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)
            attn = attn.transpose(0, 2, 1, 3)  # [B*S, T, H, D]

            # rearrange(attn, '(B S) T H D -> B (T S) (H D)', S=S)
            attn = attn.reshape(B_orig, S, attn.shape[1], H, D_key)
            attn = attn.transpose(0, 2, 1, 3, 4)
            attn = attn.reshape(B_orig, attn.shape[1] * S, H * D_key)

            attn = self.proj_keyboard(attn)
            hidden_states = hidden_states + attn

        return hidden_states
