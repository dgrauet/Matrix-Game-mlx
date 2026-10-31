"""Optional smeltr GPU-time scopes: inert unless SMELTR_AUTOLOAD=1 and smeltr imports."""
import contextlib
import os


def scope(name: str):
    """Return smeltr.scope(name) when smeltr is attached, else a no-op context manager. Never raises."""
    if os.environ.get("SMELTR_AUTOLOAD") != "1":
        return contextlib.nullcontext()
    try:
        import smeltr
        return smeltr.scope(name)
    except Exception:  # noqa: BLE001 - observability must never break inference
        return contextlib.nullcontext()
