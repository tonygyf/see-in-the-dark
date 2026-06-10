import random
from pathlib import Path

import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    requested_device = cfg.get("device", "cuda")
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[Device] {device}, torch={torch.__version__}, cuda={torch.version.cuda}")
    if requested_device == "cuda" and not use_cuda:
        print("[Warn] Requested cuda but unavailable, fallback to cpu.")
    return device


def resolve_experiment(cfg: dict) -> tuple[str, Path]:
    exp_cfg = cfg.get("experiment", {})
    exp_name = str(exp_cfg.get("name", "default"))
    out_dir = Path(exp_cfg.get("out_dir", f"runs/experiments/{exp_name}"))
    return exp_name, out_dir
