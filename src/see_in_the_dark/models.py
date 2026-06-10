import torch
import torch.nn as nn


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


def build_model(cfg: dict) -> tuple[nn.Module, dict[str, bool]]:
    model_cfg = cfg["model"]
    module_cfg = cfg.get("modules", {})
    enable_scm = bool(module_cfg.get("enable_scm", False))
    enable_dsf = bool(module_cfg.get("enable_dsf", False))
    enable_tsr = bool(module_cfg.get("enable_tsr", False))

    model = TinySegNet(
        in_channels=model_cfg["in_channels"],
        base_channels=model_cfg["base_channels"],
        num_classes=model_cfg["num_classes"],
        enable_scm=enable_scm,
        enable_dsf=enable_dsf,
        enable_tsr=enable_tsr,
    )
    flags = {
        "enable_scm": enable_scm,
        "enable_dsf": enable_dsf,
        "enable_tsr": enable_tsr,
    }
    return model, flags


def maybe_prepare_qat(model: nn.Module, cfg: dict) -> nn.Module:
    quant_cfg = cfg.get("quantization", {})
    if not quant_cfg.get("enable_qat", False):
        return model
    backend = quant_cfg.get("backend", "fbgemm")
    torch.backends.quantized.engine = backend
    model.train()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    return torch.ao.quantization.prepare_qat(model)
