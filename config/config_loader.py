from pathlib import Path
import yaml
from types import SimpleNamespace

def _dict_to_namespace(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns

def load_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = _dict_to_namespace(raw)
    for d in [cfg.paths.processed_data, cfg.paths.labels,
              cfg.paths.checkpoints, cfg.paths.logs]:
        Path(d).mkdir(parents=True, exist_ok=True)
    return cfg
