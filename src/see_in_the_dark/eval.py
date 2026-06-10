import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


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
    p_tsr = tsr_shaping(p_bin) if enable_tsr else None

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
