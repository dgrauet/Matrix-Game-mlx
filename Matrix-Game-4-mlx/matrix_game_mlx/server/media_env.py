"""macOS media environment for online play: the defaults upstream's scripts/serve.sh exports."""
import os
import shutil
import subprocess
import sys

_GI_CHECK = "import gi, numpy; gi.require_version('Gst', '1.0')"
_CANDIDATES = (lambda: sys.executable, lambda: "/opt/homebrew/bin/python3", lambda: shutil.which("python3"))


def default_sidecar_python() -> str:
    """First interpreter that can import gi + numpy and load GStreamer 1.0 (the sidecar's requirements)."""
    tried = []
    for candidate in _CANDIDATES:
        python = candidate()
        if not python or python in tried:
            continue
        tried.append(python)
        try:
            ok = subprocess.run([python, "-c", _GI_CHECK], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=30).returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if ok:
            return python
    raise RuntimeError("no Python with gi (PyGObject), numpy and GStreamer 1.0 found (tried: "
                       f"{', '.join(tried) or 'none'}); install with `brew install gstreamer pygobject3` "
                       "or set STUDIO_WEBRTC_PYTHON to a GI-enabled interpreter")


def configure_media_environment(environ=os.environ) -> None:
    """Apply scripts/serve.sh's defaults; explicit values win, except the frame pacing which serve.sh forces."""
    environ.setdefault("OMP_NUM_THREADS", "8")
    environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if not environ.get("STUDIO_WEBRTC_PYTHON"):
        environ["STUDIO_WEBRTC_PYTHON"] = default_sidecar_python()
    environ.setdefault("STUDIO_WEBRTC_RTSP_PORT", "8555")
    environ.setdefault("STUDIO_MEDIA_MAXRATE_KBPS", "2500")
    environ["STUDIO_WEBRTC_FRAME_PACING"] = "clock"
