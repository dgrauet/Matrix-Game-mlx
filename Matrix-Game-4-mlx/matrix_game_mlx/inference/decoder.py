"""Incremental Wan VAE decoding with persistent causal features (MLX port of matrix_game/inference/decoder.py)."""
import mlx.core as mx


class IncrementalDecoder:
    """Per-latent Wan2.2 decoding; caches persist across calls, so chunked output equals a one-shot decode prefix."""

    def __init__(self, vae_wrapper):
        from matrix_game_mlx.inference.wan.modules.vae2_2 import count_conv3d
        self.vae = vae_wrapper
        self.m = vae_wrapper.model
        self.feat = [None] * count_conv3d(self.m.decoder)
        self.first = True

    def push(self, z: mx.array) -> mx.array:
        """z [1, n, 48, h, w] normalized latents -> new frames [T, 3, H, W] in [-1, 1] (float32)."""
        from matrix_game_mlx.inference.wan.modules.vae2_2 import unpatchify
        mean, inv_std = self.vae._scale(self.vae.dtype)
        z = z[0].transpose(0, 2, 3, 1)[None].astype(self.vae.dtype)          # [1, n, h, w, 48]
        z = z / inv_std + mean
        x = self.m.conv2(z)
        outs = []
        for i in range(x.shape[1]):
            outs.append(self.m.decoder(x[:, i:i + 1], feat_cache=self.feat, feat_idx=[0], first_chunk=self.first))
            self.first = False
        out = unpatchify(mx.concatenate(outs, axis=1), patch_size=2)        # [1, T, H, W, 3]
        return mx.clip(out.astype(mx.float32), -1, 1)[0].transpose(0, 3, 1, 2)
