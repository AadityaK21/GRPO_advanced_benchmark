"""Load a run's configuration: configs/base.yaml, then the method's file, then command-line overrides.

    cfg = load_config("dr_grpo", ["budget.group_size=4", "optimizer.lr=3e-6"])

Rules, so that conditions stay matched:
- A method file may set only its `advantage`, `loss` and `filter` sections (filter: DAPO's dynamic sampling). Anything else it needs to change
  is a deviation from the shared setup: it goes in the paper, and here as an explicit override.
- Every key must already exist in base.yaml, except under advantage.params. A typo is an error,
  not a silently ignored setting.
- Overrides (--set key=value) may change anything; the run records them with its config.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
METHOD_SECTIONS = {"advantage", "loss", "filter"}


def load_config(method: str, overrides: list[str] | tuple = (), config_dir: Path | str = CONFIG_DIR) -> dict:
    config_dir = Path(config_dir)
    cfg = _read(config_dir / "base.yaml")
    method_file = config_dir / "methods" / f"{method}.yaml"
    if not method_file.exists():
        known = sorted(p.stem for p in (config_dir / "methods").glob("*.yaml"))
        raise ValueError(f"no method config {method_file}; known methods: {known}")
    method_cfg = _read(method_file)
    extra = set(method_cfg) - METHOD_SECTIONS
    if extra:
        raise ValueError(f"{method_file.name} may set only {sorted(METHOD_SECTIONS)}, not {sorted(extra)}: "
                         "anything else would make the comparison unmatched")
    _merge(cfg, method_cfg, where=method_file.name)
    cfg["method"] = method
    for item in overrides:
        key, value = parse_override(item)
        _set(cfg, key, value)
    validate(cfg)
    return cfg


def parse_override(item: str) -> tuple[str, object]:
    """'budget.group_size=4' -> ('budget.group_size', 4). The value is read as YAML: 4, 3e-6, true, null, [1, 2]."""
    if "=" not in item:
        raise ValueError(f"override {item!r} must look like key.path=value")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw) if raw.strip() else ""
    if isinstance(value, str) and _looks_numeric(value):   # YAML 1.1 reads '1e-5' as a string
        value = float(value)
    return key.strip(), value


def validate(cfg: dict) -> None:
    b = cfg["budget"]
    if b["group_size"] < 2:
        raise ValueError("budget.group_size must be at least 2")
    if b["rollouts_per_step"] < b["group_size"]:
        raise ValueError("budget.rollouts_per_step must be at least one group")
    if b["total_rollouts"] < b["rollouts_per_step"]:
        raise ValueError("budget.total_rollouts must cover at least one step")
    if cfg["loss"]["loss_type"] not in ("grpo", "bnpo", "dapo", "dr_grpo"):
        raise ValueError(f"unknown loss.loss_type {cfg['loss']['loss_type']!r}")
    if cfg["prompt"]["shots"] < 0 or cfg["data"]["val_size"] < 0:
        raise ValueError("prompt.shots and data.val_size must be >= 0")


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False)


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    return _fix_numbers(data)


def _merge(base: dict, new: dict, where: str, prefix: str = "") -> None:
    for key, value in new.items():
        path = f"{prefix}{key}"
        if key not in base and not path.startswith("advantage.params"):
            raise KeyError(f"{where}: unknown key {path!r} (not in base.yaml)")
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value, where, prefix=f"{path}.")
        else:
            base[key] = copy.deepcopy(value)


def _set(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    node = cfg
    for i, part in enumerate(parts[:-1]):
        path = ".".join(parts[: i + 1])
        if part not in node:
            if path.startswith("advantage.params"):
                node[part] = {}
            else:
                raise KeyError(f"override: unknown key {dotted!r} (no {path!r} in base.yaml)")
        node = node[part]
        if not isinstance(node, dict):
            raise KeyError(f"override: {path!r} is a value, not a section")
    last = parts[-1]
    if last not in node and not dotted.startswith("advantage.params."):
        raise KeyError(f"override: unknown key {dotted!r} (not in base.yaml)")
    node[last] = value


def _looks_numeric(s: str) -> bool:
    try:
        float(s)
    except ValueError:
        return False
    return True


def _fix_numbers(x):
    if isinstance(x, dict):
        return {k: _fix_numbers(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_fix_numbers(v) for v in x]
    if isinstance(x, str) and _looks_numeric(x) and x.strip().lower() not in ("nan", "inf", "-inf", "infinity"):
        return float(x)
    return x
