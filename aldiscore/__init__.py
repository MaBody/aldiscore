from pathlib import Path
import yaml
import os
from typing import List


ROOT = Path(__file__).parent
RSTATE = 0


def get_from_config(*keys: str, none_ok: bool = False):
    config = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    if os.path.exists(ROOT / "configs" / "config_local.yaml"):
        config_local = yaml.safe_load(open(ROOT / "configs" / "config_local.yaml"))
    else:
        config_local = {}

    val = _get_val_or_none(config_local, keys)
    if val is None:
        val = _get_val_or_none(config, keys)

    if not none_ok:
        assert (
            val is not None
        ), f"Keys {keys} not present in config.yaml or config_local.yaml!"
    return val


def _get_val_or_none(config: dict, keys: List[str]):
    out = config
    for key in keys:
        if (out is None) or (not isinstance(out, dict)):
            return None
        out = out.get(key, None)
    return out
