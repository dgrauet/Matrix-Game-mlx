from easydict import EasyDict
from .shared_config import wan_shared_cfg

matrix_game3 = EasyDict(__name__='Config: Matrix Game 3.0')
matrix_game3.update(wan_shared_cfg)
matrix_game3.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
matrix_game3.t5_tokenizer = 'google/umt5-xxl'
matrix_game3.vae_checkpoint = 'Wan2.2_VAE.pth'
matrix_game3.vae_stride = (4, 16, 16)
matrix_game3.patch_size = (1, 2, 2)
matrix_game3.in_dim = 48
matrix_game3.out_dim = 48
matrix_game3.dim = 3072
matrix_game3.ffn_dim = 14336
matrix_game3.freq_dim = 256
matrix_game3.num_heads = 24
matrix_game3.num_layers = 30
matrix_game3.window_size = (-1, -1)
matrix_game3.qk_norm = True
matrix_game3.cross_attn_norm = True
matrix_game3.eps = 1e-6
matrix_game3.sigma_theta = 0.8
matrix_game3.action_config = {
    'blocks': list(range(15)),
    'enable_keyboard': True,
    'enable_mouse': True,
    'heads_num': 16,
    'hidden_size': 128,
    'img_hidden_size': 3072,
    'keyboard_dim_in': 6,
    'keyboard_hidden_dim': 1024,
    'mouse_dim_in': 2,
    'mouse_hidden_dim': 1024,
    'mouse_qk_dim_list': [8, 28, 28],
    'patch_size': [1, 2, 2],
    'qk_norm': True,
    'qkv_bias': False,
    'rope_dim_list': [8, 28, 28],
    'rope_theta': 256,
    'vae_time_compression_ratio': 4,
    'windows_size': 3,
}
matrix_game3.use_text_crossattn = True
matrix_game3.sample_shift = 5.0
matrix_game3.num_inference_steps = 50
matrix_game3.sample_guide_scale = 5.0
matrix_game3.sample_neg_prompt = 'Vibrant colors, overexposure, static, blurred details, subtitles, style, artwork, painting, still image, overall grayness, worst quality, low quality, JPEG compression residue, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background, three legs, crowded background, walking backwards'
