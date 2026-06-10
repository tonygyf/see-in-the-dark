import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


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


def build_train_loader(cfg: dict, batch_size: int, image_size: int) -> DataLoader | None:
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
