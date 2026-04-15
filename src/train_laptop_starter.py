import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml


class TinySegNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_classes: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cfg(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_prepare_qat(model: nn.Module, cfg: dict) -> nn.Module:
    quant_cfg = cfg.get("quantization", {})
    if not quant_cfg.get("enable_qat", False):
        return model
    backend = quant_cfg.get("backend", "fbgemm")
    torch.backends.quantized.engine = backend
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    return torch.ao.quantization.prepare_qat(model)


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

    requested_device = cfg.get("device", "cuda")
    use_cuda = requested_device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[Device] {device}, torch={torch.__version__}, cuda={torch.version.cuda}")
    if requested_device == "cuda" and not use_cuda:
        print("[Warn] Requested cuda but unavailable, fallback to cpu.")

    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    model = TinySegNet(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        num_classes=model_cfg["num_classes"],
    ).to(device)
    model = maybe_prepare_qat(model, cfg)

    if train_cfg.get("channels_last", False) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]))
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True) and device.type == "cuda"))

    epochs = int(train_cfg["epochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    batch_size = int(train_cfg["batch_size"])
    image_size = int(train_cfg["image_size"])
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    clip_grad_norm = float(train_cfg.get("clip_grad_norm", 1.0))

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, steps_per_epoch + 1):
            x = torch.randn(batch_size, 3, image_size, image_size, device=device)
            y = torch.randint(0, 2, (batch_size, 1, image_size, image_size), device=device).float()
            if train_cfg.get("channels_last", False) and device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)

            with torch.amp.autocast("cuda", enabled=bool(train_cfg.get("amp", True) and device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y) / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum_steps
            global_step += 1

            if step % 20 == 0:
                max_mem = 0.0
                if device.type == "cuda":
                    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                print(
                    f"[Epoch {epoch}/{epochs}] step={step}/{steps_per_epoch} "
                    f"loss={running_loss / step:.4f} max_mem={max_mem:.2f}GB"
                )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "starter_last.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt)
    print(f"[Done] Saved checkpoint to: {ckpt}")


if __name__ == "__main__":
    main()
