import argparse
from pathlib import Path

import torch

from .eval import save_training_artifacts
from .models import build_model, maybe_prepare_qat
from .train import train_model
from .utils import load_cfg, resolve_device, resolve_experiment, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/laptop_4060_quant_friendly.yaml"),
        help="Path to yaml config",
    )
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg.get("seed", 42)))

    device = resolve_device(cfg)
    exp_name, out_dir = resolve_experiment(cfg)

    model, module_flags = build_model(cfg)
    print(
        "[Modules] "
        f"SCM={module_flags['enable_scm']}, "
        f"DSF={module_flags['enable_dsf']}, "
        f"TSR={module_flags['enable_tsr']}"
    )
    print(f"[Exp] name={exp_name}, out_dir={out_dir}")

    model = model.to(device)
    model = maybe_prepare_qat(model, cfg)

    result = train_model(
        model=model,
        cfg=cfg,
        device=device,
        module_flags=module_flags,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "starter_last.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt)
    save_training_artifacts(
        out_dir=out_dir,
        loss_history=result.loss_history,
        vis_triplet=result.vis_triplet,
        enable_tsr=module_flags["enable_tsr"],
    )
    print(f"[Done] Saved checkpoint to: {ckpt}")
    print(f"[Done] Saved loss csv to: {out_dir / 'train_loss.csv'}")
    print(f"[Done] Saved loss curve to: {out_dir / 'train_loss_curve.png'}")
    print(f"[Done] Saved prediction sample to: {out_dir / 'prediction_sample.png'}")


if __name__ == "__main__":
    main()
