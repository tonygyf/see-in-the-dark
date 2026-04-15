import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset


def make_center_target(mask: torch.Tensor, ksize: int = 9) -> torch.Tensor:
    pad = ksize // 2
    inv = 1.0 - mask
    eroded_inv = F.max_pool2d(inv, kernel_size=ksize, stride=1, padding=pad)
    center = 1.0 - eroded_inv
    return center.clamp(0.0, 1.0)


def tsr_shaping(binary_mask: np.ndarray) -> np.ndarray:
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask_u8)
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(out, box, 255)
    return (out > 127).astype(np.float32)


class DSFBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.regular = nn.Conv2d(channels, channels, 3, padding=1)
        self.snake = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 5), padding=(0, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, (5, 1), padding=(2, 0)),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, 2, 1),
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_reg = self.regular(x)
        v_snake = self.snake(x)
        v_cat = torch.cat([v_reg, v_snake], dim=1)
        gate_logits = self.gate(v_cat)
        gate = torch.softmax(gate_logits, dim=1)
        out = gate[:, 0:1] * v_reg + gate[:, 1:2] * v_snake
        return self.fuse(torch.cat([out, x], dim=1))


class TinySegNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_classes: int = 1,
        enable_scm: bool = False,
        enable_dsf: bool = False,
        enable_tsr: bool = False,
    ):
        super().__init__()
        self.enable_scm = enable_scm
        self.enable_dsf = enable_dsf
        self.enable_tsr = enable_tsr

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        if self.enable_dsf:
            self.dsf = DSFBlock(base_channels * 2)
        self.up = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_act = nn.ReLU(inplace=True)
        self.main_head = nn.Conv2d(base_channels, num_classes, 1)
        if self.enable_scm:
            self.scm_head = nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_channels, base_channels // 2, 2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_channels // 2, num_classes, 1),
            )
        if self.enable_tsr:
            self.center_head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encoder(x)
        if self.enable_dsf:
            enc = self.dsf(enc)
        dec = self.up_act(self.up(enc))
        logits = self.main_head(dec)
        out: dict[str, torch.Tensor] = {"logits": logits, "dec_feat": dec}
        if self.enable_scm:
            out["aux_logits"] = self.scm_head(enc)
        if self.enable_tsr:
            out["center_logits"] = self.center_head(dec)
        return out


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


def save_training_artifacts(
    out_dir: Path,
    loss_history: list[float],
    vis_triplet: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    enable_tsr: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "train_loss.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for i, loss_val in enumerate(loss_history, start=1):
            writer.writerow([i, f"{loss_val:.6f}"])

    curve_path = out_dir / "train_loss_curve.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o", linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=160)
    plt.close()

    if vis_triplet is None:
        return

    x_vis, y_vis, pred_vis = vis_triplet
    x_np = x_vis.detach().cpu().numpy().transpose(1, 2, 0)
    y_np = y_vis.detach().cpu().numpy().squeeze(0)
    p_np = pred_vis.detach().cpu().numpy().squeeze(0)
    p_bin = (p_np > 0.5).astype(np.float32)
    if enable_tsr:
        p_tsr = tsr_shaping(p_bin)
    else:
        p_tsr = None

    x_np = np.clip(x_np, 0.0, 1.0)
    y_np = np.clip(y_np, 0.0, 1.0)
    p_np = np.clip(p_np, 0.0, 1.0)

    ncols = 5 if p_tsr is not None else 4
    fig, axes = plt.subplots(1, ncols, figsize=(16, 4))
    axes[0].imshow(x_np)
    axes[0].set_title("Input")
    axes[1].imshow(y_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("GT Mask")
    axes[2].imshow(p_np, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Pred Prob")
    axes[3].imshow(p_bin, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Pred Binary")
    if p_tsr is not None:
        axes[4].imshow(p_tsr, cmap="gray", vmin=0, vmax=1)
        axes[4].set_title("TSR Shape")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    vis_path = out_dir / "prediction_sample.png"
    plt.savefig(vis_path, dpi=160)
    plt.close(fig)


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
    module_cfg = cfg.get("modules", {})
    enable_scm = bool(module_cfg.get("enable_scm", False))
    enable_dsf = bool(module_cfg.get("enable_dsf", False))
    enable_tsr = bool(module_cfg.get("enable_tsr", False))
    loss_cfg = cfg.get("loss", {})
    w_lsr = float(loss_cfg.get("w_lsr", 0.3))
    w_lss = float(loss_cfg.get("w_lss", 0.2))
    w_tsr_center = float(loss_cfg.get("w_tsr_center", 0.25))
    exp_cfg = cfg.get("experiment", {})
    exp_name = str(exp_cfg.get("name", "default"))
    out_dir = Path(exp_cfg.get("out_dir", f"runs/experiments/{exp_name}"))
    print(f"[Exp] name={exp_name}, out_dir={out_dir}")
    print(f"[Modules] SCM={enable_scm}, DSF={enable_dsf}, TSR={enable_tsr}")

    model = TinySegNet(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        num_classes=model_cfg["num_classes"],
        enable_scm=enable_scm,
        enable_dsf=enable_dsf,
        enable_tsr=enable_tsr,
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
    loss_history: list[float] = []
    vis_triplet: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
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
                outputs = model(x)
                logits = outputs["logits"]
                main_loss = criterion(logits, y)
                total_loss = main_loss
                if enable_scm and "aux_logits" in outputs:
                    aux_logits = outputs["aux_logits"]
                    lsr = criterion(aux_logits, y)
                    with torch.no_grad():
                        main_prob = torch.sigmoid(logits.detach())
                    lss = F.mse_loss(torch.sigmoid(aux_logits), main_prob)
                    total_loss = total_loss + w_lsr * lsr + w_lss * lss
                if enable_tsr and "center_logits" in outputs:
                    center_target = make_center_target(y)
                    center_loss = criterion(outputs["center_logits"], center_target)
                    total_loss = total_loss + w_tsr_center * center_loss
                loss = total_loss / grad_accum_steps

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum_steps
            loss_history.append(loss.item() * grad_accum_steps)
            global_step += 1

            if step % log_interval == 0:
                max_mem = 0.0
                if device.type == "cuda":
                    max_mem = torch.cuda.max_memory_allocated() / (1024**3)
                print(
                    f"[Epoch {epoch}/{epochs}] step={step}/{steps_per_epoch} "
                    f"loss={running_loss / step:.4f} max_mem={max_mem:.2f}GB"
                )

            if step == steps_per_epoch:
                with torch.no_grad():
                    pred_prob = torch.sigmoid(logits[:1])
                    vis_triplet = (
                        x[:1].detach()[0],
                        y[:1].detach()[0],
                        pred_prob[0],
                    )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "starter_last.pt"
    torch.save({"model": model.state_dict(), "config": cfg}, ckpt)
    save_training_artifacts(
        out_dir=out_dir,
        loss_history=loss_history,
        vis_triplet=vis_triplet,
        enable_tsr=enable_tsr,
    )
    print(f"[Done] Saved checkpoint to: {ckpt}")
    print(f"[Done] Saved loss csv to: {out_dir / 'train_loss.csv'}")
    print(f"[Done] Saved loss curve to: {out_dir / 'train_loss_curve.png'}")
    print(f"[Done] Saved prediction sample to: {out_dir / 'prediction_sample.png'}")


if __name__ == "__main__":
    main()
