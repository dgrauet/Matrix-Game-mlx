import contextlib
import sys
import types

from matrix_game_mlx.inference.scopes import scope


def test_scope_inert_without_env(monkeypatch):
    monkeypatch.delenv("SMELTR_AUTOLOAD", raising=False)
    monkeypatch.delitem(sys.modules, "smeltr", raising=False)
    assert isinstance(scope("x"), contextlib.nullcontext)
    assert "smeltr" not in sys.modules


def test_scope_uses_smeltr_when_attached(monkeypatch):
    names = []

    @contextlib.contextmanager
    def fake_scope(name):
        names.append(name)
        yield

    monkeypatch.setenv("SMELTR_AUTOLOAD", "1")
    monkeypatch.setitem(sys.modules, "smeltr", types.SimpleNamespace(scope=fake_scope))
    with scope("denoise"):
        pass
    assert names == ["denoise"]


def test_scope_survives_broken_smeltr(monkeypatch):
    def broken(name):
        raise RuntimeError("boom")

    monkeypatch.setenv("SMELTR_AUTOLOAD", "1")
    monkeypatch.setitem(sys.modules, "smeltr", types.SimpleNamespace(scope=broken))
    assert isinstance(scope("x"), contextlib.nullcontext)
