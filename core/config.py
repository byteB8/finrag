"""TOML configs with inheritance: a config may name a `base` config and override only what differs."""
from __future__ import annotations

import tomllib

from core import env


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path) -> dict:
    """`base` is relative to the project root."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if "base" in cfg:
        cfg = merge(load_cfg(env.BASE / cfg.pop("base")), cfg)
    return cfg
