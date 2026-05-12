from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Registry:
    """A tiny config-driven registry inspired by MMCV, without external deps."""

    def __init__(self, name: str):
        self.name = str(name)
        self._modules: dict[str, Callable[..., Any]] = {}

    def __contains__(self, key: str) -> bool:
        return str(key).lower() in self._modules

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._modules))

    def register(
        self,
        name: str | None = None,
        module: Callable[..., Any] | None = None,
    ):
        if module is not None:
            return self._register(module, name)

        def decorator(obj: Callable[..., Any]):
            self._register(obj, name)
            return obj

        return decorator

    def _register(self, obj: Callable[..., Any], name: str | None = None):
        key = str(name or getattr(obj, "__name__", obj.__class__.__name__)).lower()
        if key in self._modules:
            raise KeyError(f"{key!r} is already registered in {self.name}.")
        self._modules[key] = obj
        return obj

    def get(self, key: str) -> Callable[..., Any]:
        normalized = str(key).lower()
        if normalized not in self._modules:
            available = ", ".join(self.keys) or "<empty>"
            raise KeyError(
                f"{normalized!r} is not registered in {self.name}. "
                f"Available: {available}"
            )
        return self._modules[normalized]

    def build(self, cfg: Any, **kwargs):
        key = self._resolve_key(cfg)
        return self.get(key)(cfg, **kwargs)

    @staticmethod
    def _resolve_key(cfg: Any) -> str:
        if isinstance(cfg, str):
            return cfg

        model_cfg = None
        if hasattr(cfg, "get"):
            model_cfg = cfg.get("model", None)
        if model_cfg is not None and hasattr(model_cfg, "get"):
            key = model_cfg.get("name", None) or model_cfg.get("type", None)
            if key is not None:
                return str(key)

        if hasattr(cfg, "get"):
            key = cfg.get("name", None) or cfg.get("type", None)
            if key is not None:
                return str(key)

        raise KeyError("Registry.build expected cfg with a 'name' or 'type' field.")
