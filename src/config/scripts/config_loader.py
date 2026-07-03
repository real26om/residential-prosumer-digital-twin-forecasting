from pathlib import Path
from types import MappingProxyType
from typing import Any, Union
import yaml


def _deep_freeze(obj: Any) -> Any:
    """Recursively freeze dict/list so config can't be mutated accidentally."""
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_deep_freeze(x) for x in obj)
    return obj


def load_frozen_config(path: Union[str, Path]) -> MappingProxyType:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict) or "config" not in raw:
        raise ValueError("config.yml must parse to a dict with top-level key: 'config'")

    return _deep_freeze(raw)  # immutable view


def cfg_get(cfg: MappingProxyType, dotted: str) -> Any:
    """Convenience getter: cfg_get(cfg, 'config.solar.library_version')"""
    cur: Any = cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur