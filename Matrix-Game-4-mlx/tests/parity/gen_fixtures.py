"""Generate PyTorch reference fixtures with the real upstream model (quality preset; ancient-egypt, castle, racing).

Run manually from Matrix-Game-4-mlx/ (not collected by pytest):
    MATRIX_GAME_4_MLX_WEIGHTS_DIR=... python -m tests.parity.gen_fixtures --device auto [--scene castle] [--long]

Writes `$MG4_FIXTURES_DIR/<scene>_quality/` (egypt_quality, castle_quality, racing_quality): inputs.npz,
chunk_{i}.npz, floor_chunk_{i}.npz, calls_chunk{i}.npz, blocks_<call>.npz (blocks_chunk0_call0.npz unless chunk 0
has no denoising call), meta.json (and long_*.npz plus calls_long22.npz with --long). `--floor-only`,
`--calls-only`, `--call-floors-only` and `--blocks-only` regenerate only those parts from the saved fixtures.
"""
import argparse
import contextlib
import gc
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from matrix_game_mlx.scenes import scene_sink_policy
from tests.conftest import UPSTREAM_ROOT, fixtures_dir, psnr, rel_l2, to_np

SRC = Path("/Users/dgrauet/Work/mlx/tools/mlx-forge/models/matrix-game-4.0-mlx-src")
CONFIG_DIR = SRC / "wan2.2-fun-5b-control"
VAE_REPO, VAE_FILE, VAE_REVISION = "alibaba-pai/Wan2.2-Fun-5B-Control", "Wan2.2_VAE.pth", \
    "b8bc1a65ab71d054ba4636dc0dac104aa4df2686"
UPSTREAM_COMMIT = "5446525"
LATENT_SHAPE = (1, 1, 48, 30, 54)
STEPS = (1000, 500)
PAN_STEP = 8
LONG_CALL_CHUNK = 22    # position 23 > 21-frame window: rolling eviction (and fixed-sink re-RoPE for kv_only/fixed) run
KV_LAYERS = (0, 29)


@dataclass(frozen=True)
class SceneSpec:
    """One scene's fixture set: scene name, output dir, torch checkpoint dir, and the upstream sink policy."""
    scene: str
    fixture_dir: str
    src_dir: str

    @property
    def sink_mode(self) -> str:
        return scene_sink_policy(self.scene)[0]

    @property
    def sink_rope_mode(self) -> str:
        return scene_sink_policy(self.scene)[1]

    @property
    def checkpoint(self) -> Path:
        return SRC / self.src_dir / "model.safetensors"

    @property
    def sink_image(self) -> Path:
        return SRC / self.src_dir / "sink.png"


SCENE_SPECS = {s.scene: s for s in (SceneSpec("ancient-egypt", "egypt_quality", "fps-egypt"),
                                    SceneSpec("castle", "castle_quality", "fps-castle"),
                                    SceneSpec("racing", "racing_quality", "driving_game"))}
DEFAULT_SCENE = "ancient-egypt"


def n_denoising_calls(spec: SceneSpec, chunk: int) -> int:
    """Control-conditioned calls of a chunk: first_frame chunk 0 reuses the sink latent as pred (no denoising)."""
    return 0 if spec.sink_mode == "first_frame" and chunk == 0 else len(STEPS)   # first_steps == STEPS (make_stream)


def n_renoise(spec: SceneSpec, chunk: int) -> int:
    """Renoise draws of a chunk: upstream draws one between consecutive denoising steps only."""
    return max(n_denoising_calls(spec, chunk) - 1, 0)


def first_denoising_call(n_calls: list, drops: dict) -> tuple:
    """(fixture name, call index) of the first control-conditioned call; n_calls: [(name, call count)] in chunk
    order, drops: name -> per-call drop_control flags."""
    for name, count in n_calls:
        for j in range(count):
            if not drops[name][j]:
                return name, j
    raise RuntimeError("No control-conditioned call recorded")


def spec_of(meta: dict) -> SceneSpec:
    return SCENE_SPECS[meta["scene"]]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pan_frames(spec: SceneSpec, n: int) -> np.ndarray:
    """Deterministic pan over the sink image: 480x864 crops of the native frame, shifted 8 px per frame
    (ping-pong when the window reaches the image edge); a center crop, so for first_frame scenes (sink fitted to the
    frame) frame 0 is not the sink framing. Returns float32 [n, 3, 480, 864] in [-1, 1]."""
    from PIL import Image
    with Image.open(spec.sink_image) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    top, span = (h - 480) // 2, w - 864
    frames = []
    for i in range(n):
        x = (i * PAN_STEP) % (2 * span)
        left = x if x <= span else 2 * span - x
        frames.append(arr[top:top + 480, left:left + 864])
    return (np.stack(frames).transpose(0, 3, 1, 2).astype(np.float32) * (2.0 / 255.0) - 1.0).astype(np.float32)


def chunk_slices(n_chunks: int):
    start = 0
    for i in range(n_chunks):
        count = 1 if i == 0 else 4
        yield start, start + count
        start += count


def mlx_prompt_embeds(spec: SceneSpec) -> np.ndarray:
    """Scene prompt through the MLX WanTextEncoder (bf16), returned as float32 [1, 512, 4096]."""
    import mlx.core as mx
    from matrix_game_mlx.inference.loader import pack_dir, resolve_shared_files
    from matrix_game_mlx.inference.wrappers import WanTextEncoder
    from matrix_game_mlx.scenes import SCENES
    shared = resolve_shared_files(pack_dir())
    text = WanTextEncoder(str(shared["t5_encoder.safetensors"]), str(shared["google/umt5-xxl"]))
    embeds = text([SCENES[spec.scene]["prompt"]])["prompt_embeds"].astype(mx.bfloat16)
    out = to_np(embeds)
    del text, embeds
    gc.collect()
    mx.clear_cache()
    return out


def mlx_sink_latent(spec: SceneSpec) -> np.ndarray:
    """MLX's own sink encoding (VAE evidence only; the stream parity feeds the torch sink to both sides)."""
    import mlx.core as mx
    from matrix_game_mlx.inference.loader import pack_dir, resolve_shared_files
    from matrix_game_mlx.inference.stream import encode_sink_image
    from matrix_game_mlx.inference.wrappers import WanVAE22Wrapper
    shared = resolve_shared_files(pack_dir())
    vae = WanVAE22Wrapper(str(shared["vae.safetensors"]), dtype=mx.bfloat16)
    out = to_np(encode_sink_image(vae, spec.sink_image, "wan", first_frame=spec.sink_mode == "first_frame"))
    del vae
    gc.collect()
    mx.clear_cache()
    return out


@contextlib.contextmanager
def injected_renoise(torch, tensors):
    """Serve prepared tensors to upstream `torch.randn` calls of the latent shape (the stream renoise draw)."""
    real, queue = torch.randn, list(tensors)
    flat = (1, *LATENT_SHAPE[2:])

    def randn(*size, **kw):
        shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
        if shape == flat and queue:
            return queue.pop(0).reshape(shape).to(device=kw.get("device"), dtype=kw.get("dtype") or torch.float32)
        return real(*size, **kw)

    torch.randn = randn
    try:
        yield queue
    finally:
        torch.randn = real


@contextlib.contextmanager
def cpu_bf16_emulation(torch, sdpa: bool = True, acc=None):
    """Run CPU bf16 GEMM/conv/attention in fp32 and round the output to bf16 (fp32-accumulate kernel semantics).

    Native CPU bf16 kernels run at ~1 GFLOPS on Apple Silicon (a 5B forward would take ~1 h); every other op,
    including the float64 RoPE / timestep / x0 islands, runs unchanged. MPS and fp32 calls are untouched.
    `sdpa=False` keeps the native bf16 SDPA kernel and `acc=torch.float64` accumulates in float64 (torch-vs-torch
    floor variants).
    """
    F = torch.nn.functional
    names = ("linear", "conv1d", "conv2d", "conv3d") + (("scaled_dot_product_attention",) if sdpa else ())
    real = {name: getattr(F, name) for name in names}

    def upcast(fn):
        def call(*args, **kw):
            if not any(torch.is_tensor(a) and a.dtype == torch.bfloat16 and a.device.type == "cpu"
                       for a in (*args, *kw.values())):
                return fn(*args, **kw)
            cast = lambda a: a.to(acc or torch.float32) if torch.is_tensor(a) and a.dtype == torch.bfloat16 else a  # noqa: E731
            return fn(*map(cast, args), **{k: cast(v) for k, v in kw.items()}).to(torch.bfloat16)
        return call

    for name in names:
        setattr(F, name, upcast(real[name]))
    try:
        yield
    finally:
        for name in names:
            setattr(F, name, real[name])


class GenRecorder:
    """Wrap the upstream generator: record pred_x0 of every denoising (control-conditioned) call and timings."""

    def __init__(self, gen):
        self.gen, self.preds, self.seconds = gen, [], []

    def __call__(self, **kw):
        t0 = time.perf_counter()
        out = self.gen(**kw)
        self.seconds.append(time.perf_counter() - t0)
        if not kw.get("drop_control"):
            self.preds.append(out[1].detach().float().cpu().numpy())
        return out

    def parameters(self):
        return self.gen.parameters()


def stream_encode(torch, vae, pixels: np.ndarray, feat_cache: list):
    """One streaming encoder step, as CastleStream.push's Wan branch: [T,3,H,W] -> normalized [1,1,48,h,w]."""
    from matrix_game.inference.wan.modules.vae2_2 import patchify
    p = next(vae.parameters())
    x = torch.from_numpy(pixels).permute(1, 0, 2, 3).unsqueeze(0).to(p.device, p.dtype)
    model = vae.model
    hidden = model.encoder(patchify(x, patch_size=2), feat_cache=feat_cache, feat_idx=[0])
    mu = model.conv1(hidden).chunk(2, dim=1)[0]
    mean, inv_std = vae._scale(p.device, p.dtype)
    return ((mu - mean.view(1, 48, 1, 1, 1)) * inv_std.view(1, 48, 1, 1, 1)).permute(0, 2, 1, 3, 4)


def vae_roundtrip(torch, vae, pixels: np.ndarray) -> np.ndarray:
    """First-chunk streaming encode + incremental decode, as CastleStream.push does (fresh caches)."""
    from matrix_game.inference.decoder import IncrementalDecoder
    from matrix_game.inference.wan.modules.vae2_2 import count_conv3d
    latent = stream_encode(torch, vae, pixels, [None] * count_conv3d(vae.model.encoder))
    return IncrementalDecoder(vae).push(latent).float().cpu().numpy()


def load_vae(torch, spec: SceneSpec, device_pref: str, pixels_0: np.ndarray, meta: dict):
    """Pick the VAE device: MPS smoke call + CPU cross-check (PSNR > 50 dB), else CPU."""
    from huggingface_hub import hf_hub_download
    from matrix_game.inference.stream import encode_sink_image
    from matrix_game.inference.wrappers import WanVAE22Wrapper
    first_frame = spec.sink_mode == "first_frame"
    vae = WanVAE22Wrapper(vae_path=hf_hub_download(VAE_REPO, VAE_FILE, revision=VAE_REVISION)).eval()
    vae = vae.requires_grad_(False)
    if device_pref in ("auto", "mps") and torch.backends.mps.is_available():
        try:
            vae = vae.to("mps", torch.bfloat16)
            t0 = time.perf_counter()
            sink = encode_sink_image(vae, spec.sink_image, "wan", first_frame=first_frame)
            mps_dec = vae_roundtrip(torch, vae, pixels_0)
            log(f"VAE MPS smoke ok ({time.perf_counter() - t0:.1f}s); cross-checking first chunk on CPU")
            vae = vae.to("cpu")
            t0 = time.perf_counter()
            cpu_dec = vae_roundtrip(torch, vae, pixels_0)
            cpu_sink = encode_sink_image(vae, spec.sink_image, "wan", first_frame=first_frame)
            meta["vae_mps_vs_cpu"] = dict(decoded_psnr=psnr(mps_dec, cpu_dec), sink_rel_l2=rel_l2(sink, cpu_sink),
                                          cpu_seconds=time.perf_counter() - t0)
            log(f"VAE MPS vs CPU: {meta['vae_mps_vs_cpu']}")
            if meta["vae_mps_vs_cpu"]["decoded_psnr"] > 50.0:
                return vae.to("mps"), "mps"
            log("VAE MPS deviates from CPU (PSNR <= 50 dB): using CPU")
            return vae, "cpu"
        except Exception as exc:  # noqa: BLE001 - any MPS failure means CPU fallback
            log(f"VAE MPS failed ({type(exc).__name__}: {exc}); falling back to CPU")
            if device_pref == "mps":
                raise
    return vae.to("cpu", torch.bfloat16), "cpu"


def ensure_accelerate(torch) -> None:
    """Provide `accelerate.init_empty_weights` (used by upstream load_wan_dit) when accelerate is not installed.

    Same mechanism as accelerate with include_buffers=False: parameters are re-registered on the meta device.
    Re-entrant: the shim has no `__spec__`, so `find_spec` would raise once it is in sys.modules.
    """
    import importlib.util
    import sys
    import types
    if "accelerate" in sys.modules or importlib.util.find_spec("accelerate") is not None:
        return

    @contextlib.contextmanager
    def init_empty_weights(include_buffers: bool = False):
        if include_buffers:
            raise NotImplementedError("shim supports include_buffers=False only")
        real = torch.nn.Module.register_parameter

        def register_parameter(module, name, param):
            real(module, name, param)
            if param is not None:
                kwargs = dict(module._parameters[name].__dict__, requires_grad=param.requires_grad)
                module._parameters[name] = type(param)(module._parameters[name].to("meta"), **kwargs)

        torch.nn.Module.register_parameter = register_parameter
        try:
            yield
        finally:
            torch.nn.Module.register_parameter = real

    sys.modules["accelerate"] = types.SimpleNamespace(init_empty_weights=init_empty_weights)


def make_stream(spec: SceneSpec, gen, vae, prompt, sink):
    from matrix_game.inference.stream import CastleStream
    return CastleStream(gen, vae, prompt, seed=0, vae_backend="wan", steps=STEPS, first_steps=STEPS,
                        sink_latent=sink, sink_mode=spec.sink_mode, sink_rope_mode=spec.sink_rope_mode)


def load_stream(torch, spec: SceneSpec, device_pref: str, vae, prompt_np: np.ndarray, sink):
    """Build the DiT, try MPS with the sink-priming forward as smoke call, else CPU."""
    from matrix_game.inference.loading import build_dit
    ensure_accelerate(torch)
    gen = build_dit(str(CONFIG_DIR), str(spec.checkpoint), is_causal=True, dtype="bfloat16", timestep_shift=5)
    gen = gen.eval().requires_grad_(False)
    if spec.sink_mode == "first_frame":
        # No sink-priming forward at construction, so there is no MPS smoke call: stay on CPU (as the egypt fixtures)
        if device_pref == "mps":
            raise RuntimeError(f"--device mps is unsupported for {spec.scene}: first_frame sink mode has no MPS smoke "
                               "call; use --device auto or cpu")
    elif device_pref in ("auto", "mps") and torch.backends.mps.is_available():
        try:
            gen = gen.to("mps")
            t0 = time.perf_counter()
            stream = make_stream(spec, gen, vae, torch.from_numpy(prompt_np).to("mps", torch.bfloat16), sink)
            log(f"DiT MPS smoke (sink priming) ok ({time.perf_counter() - t0:.1f}s)")
            return stream, "mps"
        except Exception as exc:  # noqa: BLE001 - any MPS failure means CPU fallback
            log(f"DiT MPS failed ({type(exc).__name__}: {str(exc).splitlines()[0]}); falling back to CPU")
            if device_pref == "mps":
                raise
            gen = gen.to("cpu")
            gc.collect()
            torch.mps.empty_cache()
    t0 = time.perf_counter()
    stream = make_stream(spec, gen, vae, torch.from_numpy(prompt_np).to(torch.bfloat16), sink)
    log(f"DiT CPU sink priming {time.perf_counter() - t0:.1f}s")
    return stream, "cpu"


def run_chunks(torch, spec, stream, pixels, noises, renoises, n_chunks, on_chunk):
    recorder = GenRecorder(stream.gen)
    stream.gen = recorder
    for i, (a, b) in enumerate(chunk_slices(n_chunks)):
        t0 = time.perf_counter()
        recorder.preds.clear()
        with injected_renoise(torch, [torch.from_numpy(renoises[i])][:n_renoise(spec, i)]) as left:
            chunk = stream.push(torch.from_numpy(pixels[a:b]), frame_start=a,
                                initial_noise=torch.from_numpy(noises[i]))
        if left:
            raise RuntimeError("Prepared renoise tensor was not consumed by the upstream stream")
        on_chunk(i, chunk, recorder.preds, time.perf_counter() - t0)
    return recorder.seconds


def from_bits(u: np.ndarray) -> np.ndarray:
    """uint16 bf16 payload (bf16_bits) -> float32, exactly."""
    return (u.astype(np.uint32) << 16).view(np.float32)


def written_slot(a: np.ndarray, tokens: int = 405) -> np.ndarray:
    """The latent frame a generator call writes: the last `tokens` rows of a cache's filled part [B, n, H, D]."""
    return a[:, -tokens:]


def sink_rows_written(local_end_index: int, new_tokens: int, sink_tokens: int = 405) -> range:
    """Rows of k_sink_raw a cache write of `new_tokens` ending at `local_end_index` updates. Upstream
    CausalWanSelfAttention writes the raw (pre-RoPE) k of the slot's sink part iff the slot starts inside the sink
    (one 405-token sink frame, set_attention_window(gen, 21, 1)); empty otherwise."""
    start = local_end_index - new_tokens
    return range(start, min(sink_tokens, local_end_index)) if start < sink_tokens else range(0)


def bf16_bits(t) -> np.ndarray:
    """Exact bf16 payload of a torch tensor as uint16 (decoded by from_bits); a copy, never a view of a live cache."""
    import torch
    return t.detach().to("cpu", torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16).copy()


def kv_state(kv) -> dict:
    """All-layer KV cache state (filled part, k_sink_raw, end indices) as stored in the per-call fixtures."""
    state = {}
    for i, c in enumerate(kv):
        n = int(c["local_end_index"].item())
        state.update({f"state_k{i}": bf16_bits(c["k"][:, :n]), f"state_v{i}": bf16_bits(c["v"][:, :n]),
                      f"state_ksr{i}": bf16_bits(c["k_sink_raw"]),
                      f"state_gei{i}": np.array(int(c["global_end_index"].item())), f"state_lei{i}": np.array(n)})
    return state


def load_kv_state(torch, gen, state: dict):
    """Fresh upstream caches filled from a stored KV state (inverse of kv_state)."""
    from matrix_game.inference.wrappers import init_crossattn_cache, init_kv_cache
    bits = lambda u: torch.from_numpy(u.view(np.int16)).view(torch.bfloat16)  # noqa: E731
    kv = init_kv_cache(gen, 1, 21 * 405, torch.bfloat16, "cpu")
    for i, c in enumerate(kv):
        n = int(state[f"state_lei{i}"])
        c["k"][:, :n], c["v"][:, :n] = bits(state[f"state_k{i}"]), bits(state[f"state_v{i}"])
        c["k_sink_raw"] = bits(state[f"state_ksr{i}"]).clone()
        c["global_end_index"].fill_(int(state[f"state_gei{i}"]))
        c["local_end_index"].fill_(n)
    return kv, init_crossattn_cache(gen, 1, torch.bfloat16, "cpu")


class CallRecorder:
    """Wrap the upstream generator: record each call's inputs, pred_x0 and the layer 0/29 KV after the call, plus the
    layer 0/29 k_sink_raw after the call when the call writes into the sink (`ksr{layer}`: racing's chunk-0 clean pass
    at slot 0; raw k cannot be recovered from the roped cache, so it is stored)."""

    def __init__(self, gen, kv):
        self.gen, self.kv, self.calls = gen, kv, []

    def __call__(self, **kw):
        out = self.gen(**kw)
        call = dict(x=bf16_bits(kw["noisy_image_or_video"]), t=kw["timestep"].float().cpu().numpy(),
                    start=np.array(kw["current_start"]), drop=np.array(bool(kw.get("drop_control"))),
                    pred=bf16_bits(out[1]))
        for layer in KV_LAYERS:
            c = self.kv[layer]
            n = int(c["local_end_index"].item())
            call.update({f"k{layer}": bf16_bits(c["k"][:, :n]), f"v{layer}": bf16_bits(c["v"][:, :n])})
            if sink_rows_written(n, int(c["global_end_index"].item()) - int(kw["current_start"])):
                call[f"ksr{layer}"] = bf16_bits(c["k_sink_raw"])
        self.calls.append(call)
        return out

    def parameters(self):
        return self.gen.parameters()


def record_push(torch, spec, stream, index, pixels, frame_start, noise, renoise):
    """One upstream push with every generator call recorded; returns (chunk, state before, control, calls)."""
    state = kv_state(stream.kv)
    real, recorder = stream.gen, CallRecorder(stream.gen, stream.kv)
    stream.gen = recorder
    try:
        with injected_renoise(torch, [torch.from_numpy(renoise)][:n_renoise(spec, index)]) as left:
            chunk = stream.push(torch.from_numpy(pixels), frame_start=frame_start, initial_noise=torch.from_numpy(noise))
    finally:
        stream.gen = real
    if left:
        raise RuntimeError("Prepared renoise tensor was not consumed by the upstream stream")
    return chunk, state, bf16_bits(chunk.control), recorder.calls


def save_calls(path: Path, state: dict, control: np.ndarray, calls: list) -> None:
    np.savez(path, control=control, n_calls=np.array(len(calls)), **state,
             **{f"c{j}_{k}": v for j, call in enumerate(calls) for k, v in call.items()})


FLOOR_VARIANTS = ("native_sdpa", "native_cpu_bf16", "fp64_acc")


def floor_variant(torch, name: str):
    """Context of one torch kernel variant used for the floors (see meta.json "call_floors" / "block_floors")."""
    return {"native_sdpa": lambda: cpu_bf16_emulation(torch, sdpa=False), "native_cpu_bf16": contextlib.nullcontext,
            "fp64_acc": lambda: cpu_bf16_emulation(torch, acc=torch.float64)}[name]()


def build_torch_dit(torch, spec: SceneSpec):
    """Upstream causal DiT on CPU, configured as CastleStream does (21-latent window, 1 sink, 8192 RoPE rows)."""
    from matrix_game.inference.loading import build_dit
    from matrix_game.inference.rollout import set_attention_window
    ensure_accelerate(torch)                     # after upstream imports (transformers probes accelerate)
    gen = build_dit(str(CONFIG_DIR), str(spec.checkpoint), is_causal=True, dtype="bfloat16", timestep_shift=5)
    gen = gen.eval().requires_grad_(False)
    set_attention_window(gen, 21, 1)
    gen.model.set_rope_max_len(8192)
    return gen


def call_floor(torch, spec: SceneSpec, gen, prompt, fixture) -> list:
    """Torch-vs-torch floor per call: rel L2 of pred and of the layer 0/29 KV slot it writes (and of the layer 0/29
    k_sink_raw when the call writes into the sink), for the three kernel variants."""
    out = []
    for j in range(int(fixture["n_calls"])):
        row = {}
        for name in FLOOR_VARIANTS:
            kv, xc = load_kv_state(torch, gen, fixture)
            with floor_variant(torch, name):
                _, pred = gen(noisy_image_or_video=torch.from_numpy(from_bits(fixture[f"c{j}_x"])).to(torch.bfloat16),
                              conditional_dict={"prompt_embeds": prompt,
                                                "control_latent": torch.from_numpy(from_bits(fixture["control"])).to(torch.bfloat16)},
                              timestep=torch.from_numpy(fixture[f"c{j}_t"]), kv_cache=kv, crossattn_cache=xc,
                              current_start=int(fixture[f"c{j}_start"]), drop_control=bool(fixture[f"c{j}_drop"]),
                              sink_rope_mode=spec.sink_rope_mode)
            row[f"{name}_pred"] = rel_l2(pred, from_bits(fixture[f"c{j}_pred"]))
            for layer in KV_LAYERS:
                for kind in ("k", "v"):
                    got = bf16_bits(kv[layer][kind][:, :int(kv[layer]["local_end_index"].item())])
                    row[f"{name}_{kind}{layer}"] = rel_l2(*(from_bits(written_slot(a)) for a in (got, fixture[f"c{j}_{kind}{layer}"])))
                if f"c{j}_ksr{layer}" in fixture.files:                # the call wrote into the sink
                    row[f"{name}_ksr{layer}"] = rel_l2(from_bits(bf16_bits(kv[layer]["k_sink_raw"])),
                                                       from_bits(fixture[f"c{j}_ksr{layer}"]))
        out.append(row)
    return out


def call_fixtures(out: Path) -> None:
    """Per-call fixtures: torch state at each chunk boundary (chunks 0-2 and long chunk LONG_CALL_CHUNK), every
    generator call's inputs and outputs, and the torch kernel-variant floor of each call (meta.json)."""
    import types
    import torch
    from huggingface_hub import hf_hub_download
    from matrix_game.inference.wrappers import WanVAE22Wrapper
    torch.set_grad_enabled(False)
    meta = json.loads((out / "meta.json").read_text())
    spec = spec_of(meta)
    inputs = np.load(out / "inputs.npz")
    t0 = time.perf_counter()
    vae = WanVAE22Wrapper(vae_path=hf_hub_download(VAE_REPO, VAE_FILE, revision=VAE_REVISION)).eval()
    vae = vae.requires_grad_(False).to(meta["device_vae"], torch.bfloat16)
    sink = torch.from_numpy(inputs["sink_latent"])
    names = []
    with cpu_bf16_emulation(torch):
        stream, _ = load_stream(torch, spec, meta["device_dit"], vae, inputs["prompt_embeds"], sink)
        for i, (a, b) in enumerate(chunk_slices(meta["chunks"])):
            chunk, state, control, calls = record_push(torch, spec, stream, i, inputs[f"pixels_{i}"], a,
                                                       inputs[f"initial_noise_{i}"], inputs[f"renoise_{i}"])
            if not np.array_equal(to_np(chunk.latent), np.load(out / f"chunk_{i}.npz")["pred"]):
                raise RuntimeError(f"Recorded chunk {i} does not reproduce the fixture bitwise")
            save_calls(out / f"calls_chunk{i}.npz", state, control, calls)
            names.append(f"calls_chunk{i}")
            log(f"per-call fixture chunk {i}: {len(calls)} calls")
        if (out / "long_inputs.npz").exists():
            # For first_frame scenes chunk 0 is the clean pass on the stored sink pred at current_start=0.
            # The KV state at a chunk boundary only depends on the sink priming and each chunk's clean pass (the
            # denoising writes land in the slot that pass overwrites), so replaying the stored long preds through
            # the clean pass rebuilds it; the encoder cache advances on the same pixels.
            long_inputs = np.load(out / "long_inputs.npz")
            stream = make_stream(spec, stream.gen, vae, stream.prompt, sink)
            for i, (a, b) in enumerate(chunk_slices(LONG_CALL_CHUNK)):
                stream_encode(torch, vae, long_inputs[f"pixels_{i}"], stream.encoder_features)
                pred = torch.from_numpy(np.load(out / f"long_chunk_{i}.npz")["pred"]).to(stream.dtype)
                stream.gen(noisy_image_or_video=pred, conditional_dict={"prompt_embeds": stream.prompt},
                           timestep=torch.zeros(1, 1), drop_control=True, kv_cache=stream.kv, crossattn_cache=stream.xc,
                           current_start=(i + stream.sink_offset) * stream.fsl, sink_rope_mode=stream.sink_rope_mode)
                stream.index, stream.frame = i + 1, b
            stream.decoder = types.SimpleNamespace(push=lambda z: torch.zeros(4, 3, 480, 864))   # not under test
            i = LONG_CALL_CHUNK
            chunk, state, control, calls = record_push(torch, spec, stream, i, long_inputs[f"pixels_{i}"], stream.frame,
                                                       long_inputs[f"initial_noise_{i}"], long_inputs[f"renoise_{i}"])
            if not np.array_equal(to_np(chunk.latent), np.load(out / f"long_chunk_{i}.npz")["pred"]):
                raise RuntimeError(f"Rebuilt long state does not reproduce long chunk {i} bitwise")
            save_calls(out / f"calls_long{i}.npz", state, control, calls)
            names.append(f"calls_long{i}")
            log(f"per-call fixture long chunk {i}: {len(calls)} calls")
    log(f"per-call fixtures recorded in {time.perf_counter() - t0:.0f}s")
    call_floors(out, stream.gen, stream.prompt, names)


def call_floors(out: Path, gen=None, prompt=None, names=None) -> None:
    """Torch kernel-variant floor of every per-call fixture, written to meta.json["call_floors"]."""
    import torch
    torch.set_grad_enabled(False)
    meta = json.loads((out / "meta.json").read_text())
    spec = spec_of(meta)
    if gen is None:
        gen = build_torch_dit(torch, spec)
        prompt = torch.from_numpy(np.load(out / "inputs.npz")["prompt_embeds"]).to(torch.bfloat16)
        names = sorted(p.stem for p in out.glob("calls_*.npz"))
    t0 = time.perf_counter()
    floors = {}
    for name in names:
        floors[name] = call_floor(torch, spec, gen, prompt, np.load(out / f"{name}.npz"))
        log(f"floor {name}: {floors[name]}")
    meta["call_floors"] = dict(
        variants="native_sdpa: cpu_bf16_emulation(sdpa=False); native_cpu_bf16: no emulation; "
                 "fp64_acc: cpu_bf16_emulation(acc=float64); same injected state",
        metric="rel L2 vs the fixture call (pred, and the KV slot written at layers 0/29)", calls=floors,
        seconds=time.perf_counter() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"per-call floors done in {meta['call_floors']['seconds']:.0f}s")


def block_fixtures(out: Path) -> None:
    """Per-block isolated fixture for the first control-conditioned call (chunk 0 call 0, or chunk 1 call 0 when chunk
    0 only has the clean pass; meta.json "block_call"): every block's torch input/output and the shared e0 / context
    (blocks_<call>.npz), and each block's torch kernel-variant floor with the derived gate 1.5 x max variant
    (meta.json "block_floors")."""
    import torch
    torch.set_grad_enabled(False)
    meta = json.loads((out / "meta.json").read_text())
    spec = spec_of(meta)
    fixtures = {f"calls_chunk{i}": np.load(out / f"calls_chunk{i}.npz") for i in range(meta["chunks"])}
    name, j = first_denoising_call(
        [(n, int(f["n_calls"])) for n, f in fixtures.items()],
        {n: [bool(f[f"c{k}_drop"]) for k in range(int(f["n_calls"]))] for n, f in fixtures.items()})
    fixture = fixtures[name]
    meta["block_call"] = [name, j]
    prompt = torch.from_numpy(np.load(out / "inputs.npz")["prompt_embeds"]).to(torch.bfloat16)
    t0 = time.perf_counter()
    gen = build_torch_dit(torch, spec)
    blocks = gen.model.blocks
    acts, kwargs = {}, {}

    def capture(i):
        def hook(module, args, kw):
            acts[f"x{i}"] = bf16_bits(args[0])
            kwargs[i] = dict(kw)
        return hook

    hooks = [blk.register_forward_pre_hook(capture(i), with_kwargs=True) for i, blk in enumerate(blocks)]
    hooks.append(blocks[-1].register_forward_hook(lambda module, args, y: acts.__setitem__(f"x{len(blocks)}", bf16_bits(y))))
    kv, xc = load_kv_state(torch, gen, fixture)
    call = dict(noisy_image_or_video=torch.from_numpy(from_bits(fixture[f"c{j}_x"])).to(torch.bfloat16),
                conditional_dict={"prompt_embeds": prompt,
                                  "control_latent": torch.from_numpy(from_bits(fixture["control"])).to(torch.bfloat16)},
                timestep=torch.from_numpy(fixture[f"c{j}_t"]), current_start=int(fixture[f"c{j}_start"]),
                sink_rope_mode=spec.sink_rope_mode)
    try:
        with cpu_bf16_emulation(torch):
            _, pred = gen(kv_cache=kv, crossattn_cache=xc, **call)
    finally:
        for h in hooks:
            h.remove()
    if not np.array_equal(bf16_bits(pred), fixture[f"c{j}_pred"]):
        raise RuntimeError(f"Hooked call does not reproduce {name} call {j} bitwise")
    acts["e0"], acts["context"] = bf16_bits(kwargs[0]["e"]), bf16_bits(kwargs[0]["context"])
    np.savez(out / f"blocks_{name.removeprefix('calls_')}_call{j}.npz", **acts)
    floors = []
    for i, blk in enumerate(blocks):
        x = torch.from_numpy(from_bits(acts[f"x{i}"])).to(torch.bfloat16)
        row = {}
        for variant in FLOOR_VARIANTS:
            kv, xc = load_kv_state(torch, gen, fixture)
            with floor_variant(torch, variant):
                y = blk(x, **dict(kwargs[i], kv_cache=kv[i], crossattn_cache=xc[i]))
            row[variant] = rel_l2(y, from_bits(acts[f"x{i + 1}"]))
        row["gate"] = 1.5 * max(row[variant] for variant in FLOOR_VARIANTS)
        floors.append(row)
        log(f"block {i}: {row}")
    meta["block_floors"] = dict(
        call=f"{name} call {j}", metric="rel L2 of the block output vs torch, on torch's exact block input",
        variants=list(FLOOR_VARIANTS), gate_rule="1.5 x max variant", blocks=floors, seconds=time.perf_counter() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"per-block fixture done in {meta['block_floors']['seconds']:.0f}s")


def torch_reference(args, spec: SceneSpec, out: Path, meta: dict, prompt, pixels, noises, renoises):
    """Upstream CastleStream on the chosen devices; writes inputs.npz and chunk_*.npz, returns the sink latent."""
    import torch
    torch.set_grad_enabled(False)
    meta["torch_version"] = torch.__version__
    meta["cpu_bf16"] = "emulated: fp32 linear/conv/SDPA with bf16-rounded outputs (cpu_bf16_emulation)"
    meta["repo_commit"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=UPSTREAM_ROOT,
                                         capture_output=True, text=True).stdout.strip()
    with cpu_bf16_emulation(torch):
        return _torch_run(args, spec, out, meta, prompt, pixels, noises, renoises)


def _torch_run(args, spec, out, meta, prompt, pixels, noises, renoises):
    import torch
    # UPSTREAM_ROOT is on sys.path via tests.conftest.
    log("loading torch VAE")
    vae, meta["device_vae"] = load_vae(torch, spec, args.device, pixels[:1], meta)
    from matrix_game.inference.stream import encode_sink_image
    sink = encode_sink_image(vae, spec.sink_image, "wan", first_frame=spec.sink_mode == "first_frame").float().cpu()
    log(f"VAE device: {meta['device_vae']}")
    log("loading torch DiT")
    stream, meta["device_dit"] = load_stream(torch, spec, args.device, vae, prompt, sink)
    log(f"DiT device: {meta['device_dit']}; VAE device: {meta['device_vae']}")

    inputs = dict(prompt_embeds=prompt, sink_latent=sink.numpy())
    for i, (a, b) in enumerate(chunk_slices(args.chunks)):
        inputs[f"pixels_{i}"] = pixels[a:b]
        inputs[f"initial_noise_{i}"], inputs[f"renoise_{i}"] = noises[i], renoises[i]
    np.savez(out / "inputs.npz", **inputs)
    meta["chunk_seconds"] = []

    def save_chunk(i, chunk, preds, seconds):
        if len(preds) != n_denoising_calls(spec, i):
            raise RuntimeError(f"Expected {n_denoising_calls(spec, i)} denoising calls, recorded {len(preds)}")
        step0 = dict(pred_step0=preds[0]) if preds else {}
        np.savez_compressed(out / f"chunk_{i}.npz", control=to_np(chunk.control), **step0,
                            pred=to_np(chunk.latent), decoded=to_np(chunk.pixels),
                            kv_k_layer0=to_np(stream.kv[0]["k"]), kv_k_layer29=to_np(stream.kv[29]["k"]))
        meta["chunk_seconds"].append(seconds)
        log(f"chunk {i}: {seconds:.1f}s")

    t0 = time.perf_counter()
    meta["forward_seconds"] = run_chunks(torch, spec, stream, pixels, noises, renoises, args.chunks, save_chunk)
    meta["run_seconds"] = time.perf_counter() - t0

    if args.long:
        n_long = 25
        long_pixels = pan_frames(spec, 1 + 4 * (n_long - 1))
        long_rng = np.random.default_rng(4321)
        long_noise = [long_rng.standard_normal(LATENT_SHAPE, dtype=np.float32) for _ in range(2 * n_long)]
        long_inputs = {}
        for i, (a, b) in enumerate(chunk_slices(n_long)):
            long_inputs[f"pixels_{i}"] = long_pixels[a:b]
            long_inputs[f"initial_noise_{i}"], long_inputs[f"renoise_{i}"] = long_noise[2 * i], long_noise[2 * i + 1]
        np.savez(out / "long_inputs.npz", **long_inputs)
        stream = make_stream(spec, stream.gen.gen, vae, stream.prompt, sink)

        def save_long(i, chunk, preds, seconds):
            np.savez_compressed(out / f"long_chunk_{i}.npz", pred=to_np(chunk.latent), decoded=to_np(chunk.pixels))
            log(f"long chunk {i}: {seconds:.1f}s")

        t0 = time.perf_counter()
        run_chunks(torch, spec, stream, long_pixels, long_noise[0::2], long_noise[1::2], n_long, save_long)
        meta["long_chunks"], meta["long_seconds"] = n_long, time.perf_counter() - t0

    return sink


def floor_reference(out: Path) -> None:
    """Torch-vs-torch floor for the 3-chunk set, from the saved inputs (main fixtures are not touched).

    DiT tensors: the same upstream stream with only the CPU SDPA kernel changed (native bf16 instead of
    fp32-emulated). control: the same streaming encoder on the other VAE device (the SDPA variant does not
    touch the VAE, so its control would equal the fixture bitwise).
    """
    import torch
    from huggingface_hub import hf_hub_download
    from matrix_game.inference.wan.modules.vae2_2 import count_conv3d
    from matrix_game.inference.wrappers import WanVAE22Wrapper
    torch.set_grad_enabled(False)
    meta = json.loads((out / "meta.json").read_text())
    spec = spec_of(meta)
    inputs = np.load(out / "inputs.npz")
    n_chunks = meta["chunks"]
    pixels = [inputs[f"pixels_{i}"] for i in range(n_chunks)]
    t0 = time.perf_counter()
    vae = WanVAE22Wrapper(vae_path=hf_hub_download(VAE_REPO, VAE_FILE, revision=VAE_REVISION)).eval()
    vae = vae.requires_grad_(False).to(meta["device_vae"], torch.bfloat16)
    floors = [{} for _ in range(n_chunks)]

    def save(i, chunk, preds, seconds):
        step0 = dict(pred_step0=preds[0]) if preds else {}
        floors[i].update(**step0, pred=to_np(chunk.latent), decoded=to_np(chunk.pixels),
                         kv_k_layer0=to_np(stream.kv[0]["k"]), kv_k_layer29=to_np(stream.kv[29]["k"]))
        log(f"floor chunk {i}: {seconds:.1f}s")

    with cpu_bf16_emulation(torch, sdpa=False):
        stream, device_dit = load_stream(torch, spec, meta["device_dit"], vae, inputs["prompt_embeds"],
                                         torch.from_numpy(inputs["sink_latent"]))
        if device_dit != meta["device_dit"]:
            raise RuntimeError(f"Floor DiT device {device_dit} != fixture DiT device {meta['device_dit']}")
        run_chunks(torch, spec, stream, np.concatenate(pixels), [inputs[f"initial_noise_{i}"] for i in range(n_chunks)],
                   [inputs[f"renoise_{i}"] for i in range(n_chunks)], n_chunks, save)
    del stream
    gc.collect()
    control_device = "cpu" if meta["device_vae"] == "mps" else "mps"
    vae = vae.to(control_device)
    feats = [None] * count_conv3d(vae.model.encoder)
    with cpu_bf16_emulation(torch):
        for i in range(n_chunks):
            floors[i]["control"] = stream_encode(torch, vae, pixels[i], feats).float().cpu().numpy()
    for i, tensors in enumerate(floors):
        np.savez_compressed(out / f"floor_chunk_{i}.npz", **tensors)
    meta["floor"] = dict(
        dit_variant="native bf16 CPU SDPA (cpu_bf16_emulation(sdpa=False)); everything else as the fixture run",
        control_variant=f"torch streaming VAE encoder on {control_device} (fixture: {meta['device_vae']})",
        chunks=n_chunks, seconds=time.perf_counter() - t0)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"floor fixtures done in {meta['floor']['seconds']:.0f}s")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(SCENE_SPECS), default=DEFAULT_SCENE)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--long", action="store_true", help="Also run 25 chunks (pred and decoded only)")
    parser.add_argument("--floor-only", action="store_true",
                        help="Only (re)generate floor_chunk_*.npz from the existing inputs.npz and meta.json")
    parser.add_argument("--calls-only", action="store_true",
                        help="Only (re)generate the per-call fixtures calls_*.npz from the existing fixtures")
    parser.add_argument("--call-floors-only", action="store_true",
                        help="Only (re)measure the torch kernel-variant floor of the existing per-call fixtures")
    parser.add_argument("--blocks-only", action="store_true",
                        help="Only (re)generate the per-block fixture of the first denoising call and its floors")
    return parser.parse_args(argv)


def scene_out_dir(args) -> Path:
    return fixtures_dir() / SCENE_SPECS[args.scene].fixture_dir


def main(argv=None):
    args = parse_args(argv)
    spec, out = SCENE_SPECS[args.scene], scene_out_dir(args)
    if args.floor_only:
        floor_reference(out)
        return
    if args.calls_only:
        call_fixtures(out)
        return
    if args.call_floors_only:
        call_floors(out)
        return
    if args.blocks_only:
        block_fixtures(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    meta = dict(scene=spec.scene, preset="quality", steps=list(STEPS), sink_mode=spec.sink_mode,
                sink_rope_mode=spec.sink_rope_mode, chunks=args.chunks, pan_step_px=PAN_STEP, noise_seed=1234,
                checkpoint=str(spec.checkpoint),
                upstream_commit=UPSTREAM_COMMIT, prompt_embeds_source="mlx WanTextEncoder bf16")

    log("prompt embeddings (MLX T5)")
    prompt = mlx_prompt_embeds(spec)
    n_frames = 1 + 4 * (args.chunks - 1)
    pixels = pan_frames(spec, n_frames)
    rng = np.random.default_rng(1234)
    noises, renoises = [], []
    for _ in range(args.chunks):
        noises.append(rng.standard_normal(LATENT_SHAPE, dtype=np.float32))
        renoises.append(rng.standard_normal(LATENT_SHAPE, dtype=np.float32))

    sink = torch_reference(args, spec, out, meta, prompt, pixels, noises, renoises)
    gc.collect()
    log("MLX sink encoding (VAE evidence)")
    mlx_sink = mlx_sink_latent(spec)
    meta["mlx_vs_torch_sink_rel_l2"] = rel_l2(mlx_sink, sink.numpy())
    meta["total_seconds"] = time.perf_counter() - t_start
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"done in {meta['total_seconds']:.0f}s: {out}")
    floor_reference(out)
    gc.collect()                                 # one torch DiT alive at a time
    call_fixtures(out)
    gc.collect()
    block_fixtures(out)


if __name__ == "__main__":
    main()
