from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .datasets import build_train_loader


@dataclass
class TrainResult:
    loss_history: list[float]
    vis_triplet: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


def make_center_target(mask: torch.Tensor, ksize: int = 9) -> torch.Tensor:
    pad = ksize // 2
    inv = 1.0 - mask
    eroded_inv = F.max_pool2d(inv, kernel_size=ksize, stride=1, padding=pad)
    center = 1.0 - eroded_inv
    return center.clamp(0.0, 1.0)


def train_model(
    model: nn.Module,
    cfg: dict,
    device: torch.device,
    module_flags: dict[str, bool],
) -> TrainResult:
    train_cfg = cfg["train"]
    loss_cfg = cfg.get("loss", {})
    enable_scm = module_flags["enable_scm"]
    enable_tsr = module_flags["enable_tsr"]
    w_lsr = float(loss_cfg.get("w_lsr", 0.3))
    w_lss = float(loss_cfg.get("w_lss", 0.2))
    w_tsr_center = float(loss_cfg.get("w_tsr_center", 0.25))

    if train_cfg.get("channels_last", False) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]))
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(train_cfg.get("amp", True) and device.type == "cuda"),
    )

    epochs = int(train_cfg["epochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    batch_size = int(train_cfg["batch_size"])
    image_size = int(train_cfg["image_size"])
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    clip_grad_norm = float(train_cfg.get("clip_grad_norm", 1.0))
    log_interval = int(train_cfg.get("log_interval", 20))

    loader = build_train_loader(cfg, batch_size=batch_size, image_size=image_size)
    data_iter = iter(loader) if loader is not None else None

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

            with torch.amp.autocast(
                "cuda",
                enabled=bool(train_cfg.get("amp", True) and device.type == "cuda"),
            ):
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

    return TrainResult(loss_history=loss_history, vis_triplet=vis_triplet)
