import os
import unittest
from unittest.mock import patch

from matrix_game_mlx.server import media_env
from matrix_game_mlx.server.transport.output import sidecar_environment


class SidecarTests(unittest.TestCase):
    def test_model_python_paths_do_not_cross_interpreters(self):
        with patch.dict(os.environ, {"PYTHONPATH": "model-site", "PYTHONHOME": "model-home",
                                     "PYTHONUSERBASE": "model-user", "STUDIO_WEBRTC_RTSP_PORT": "8555"}):
            env = sidecar_environment()
            for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
                self.assertNotIn(key, env)
                self.assertIn(key, os.environ)
            self.assertEqual(env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(env["STUDIO_WEBRTC_RTSP_PORT"], "8555")


class MediaEnvironmentTests(unittest.TestCase):
    def test_configure_sets_serve_sh_defaults(self):
        env = {"STUDIO_WEBRTC_PYTHON": "/x/python"}
        media_env.configure_media_environment(env)
        self.assertEqual(env, {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1",
                               "STUDIO_WEBRTC_PYTHON": "/x/python", "STUDIO_WEBRTC_RTSP_PORT": "8555",
                               "STUDIO_MEDIA_MAXRATE_KBPS": "2500", "STUDIO_WEBRTC_FRAME_PACING": "clock"})

    def test_configure_keeps_explicit_values(self):
        env = {"STUDIO_WEBRTC_PYTHON": "/x/python", "STUDIO_WEBRTC_RTSP_PORT": "9999",
               "STUDIO_WEBRTC_FRAME_PACING": "burst"}
        with patch.object(media_env, "default_sidecar_python", side_effect=AssertionError("resolved")):
            media_env.configure_media_environment(env)
        self.assertEqual(env["STUDIO_WEBRTC_PYTHON"], "/x/python")
        self.assertEqual(env["STUDIO_WEBRTC_RTSP_PORT"], "9999")
        self.assertEqual(env["STUDIO_WEBRTC_FRAME_PACING"], "clock")

    def test_default_sidecar_python_needs_gi(self):
        with patch.object(media_env.subprocess, "run", side_effect=OSError("no interpreter")):
            with self.assertRaises(RuntimeError) as ctx:
                media_env.default_sidecar_python()
        self.assertIn("STUDIO_WEBRTC_PYTHON", str(ctx.exception))
        self.assertIn("gstreamer pygobject3", str(ctx.exception))

    def test_default_sidecar_python_picks_first_working(self):
        calls = []

        def run(cmd, **kwargs):
            calls.append(cmd[0])
            return type("R", (), {"returncode": 0 if len(calls) == 2 else 1})()

        with patch.object(media_env.subprocess, "run", side_effect=run):
            self.assertEqual(media_env.default_sidecar_python(), calls[1])

    def test_sidecar_import_stays_light(self):
        import subprocess
        import sys
        from pathlib import Path
        code = ("import sys, matrix_game_mlx.server.transport.output as o; "
                "sys.exit(any(m in sys.modules for m in ('mlx', 'torch', 'aiohttp')))")
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1]).returncode, 0)


if __name__ == "__main__":
    unittest.main()
