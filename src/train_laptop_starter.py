import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset


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


class CTW1500PaddleDataset(Dataset):
    def __init__(self, imgs_root: Path, split_file: Path, image_size: int):
        self.imgs_root = imgs_root
        self.split_file = split_file
        self.image_size = image_size
        self.samples = self._load_samples()

    def _parse_line(self, line: str) -> tuple[str, list[dict]]:
        idx = line.find("[")
        if idx == -1:
            raise ValueError(f"Invalid annotation line (missing json): {line[:80]}")
        image_rel_path = line[:idx].strip()
        ann_json = line[idx:].strip()
        polygons = json.loads(ann_json)
        return image_rel_path, polygons

    def _load_samples(self) -> list[tuple[Path, list[dict]]]:
        lines = self.split_file.read_text(encoding="utf-8").splitlines()
        samples: list[tuple[Path, list[dict]]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            image_rel_path, polygons = self._parse_line(line)
            image_path = self.imgs_root / image_rel_path
            samples.append((image_path, polygons))
        if not samples:
            raise ValueError(f"No samples loaded from split file: {self.split_file}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, polygons = self.samples[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            points = np.asarray(poly.get("points", []), dtype=np.int32)
            if points.ndim != 2 or points.shape[0] < 3:
                continue
            cv2.fillPoly(mask, [points], color=1)

        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )

        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(mask).unsqueeze(0).float()
        return x, y


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


def maybe_build_ctw1500_loader(cfg: dict, batch_size: int, image_size: int) -> DataLoader | None:
    data_cfg = cfg.get("data", {})
    if not data_cfg.get("use_ctw1500_paddle_test", False):
        return None

    split_file = Path(
        data_cfg.get(
            "split_file",
            "data/raw/ctw1500/paddle_format/ctw1500/imgs/test.txt",
        )
    )
    imgs_root = Path(
        data_cfg.get(
            "imgs_root",
            "data/raw/ctw1500/paddle_format/ctw1500/imgs",
        )
    )

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not imgs_root.exists():
        raise FileNotFoundError(f"Image root not found: {imgs_root}")

    dataset = CTW1500PaddleDataset(
        imgs_root=imgs_root,
        split_file=split_file,
        image_size=image_size,
    )
    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", True))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"[Data] CTW1500 paddle test enabled, samples={len(dataset)}, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )
    return loader


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
    log_interval = int(train_cfg.get("log_interval", 20))
    loader = maybe_build_ctw1500_loader(cfg, batch_size=batch_size, image_size=image_size)
    data_iter = iter(loader) if loader is not None else None

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step in range(1, steps_per_epoch + 1):
            if loader is None:
                x = torch.randn(batch_size, 3, image_size, image_size, device=device)
                y = torch.randint(0, 2, (batch_size, 1, image_size, image_size), device=device).float()
            else:
                assert data_iter is not None
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    x, y = next(data_iter)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

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

            if step % log_interval == 0:
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
