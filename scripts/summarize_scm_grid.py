import argparse
import csv
import re
from pathlib import Path


def read_last_loss(csv_path: Path) -> float | None:
    if not csv_path.exists():
        return None
    rows = csv_path.read_text(encoding="utf-8").splitlines()
    if len(rows) <= 1:
        return None
    last = rows[-1].split(",")
    if len(last) < 2:
        return None
    return float(last[1])


def read_max_mem(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-16")
    except UnicodeError:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    vals = [float(v) for v in re.findall(r"max_mem=([0-9.]+)GB", text)]
    if not vals:
        return None
    return max(vals)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    exp_dirs = [p for p in root.iterdir() if p.is_dir()]
    exp_dirs.sort(key=lambda p: p.name)

    records: list[dict] = []
    for exp_dir in exp_dirs:
        loss = read_last_loss(exp_dir / "train_loss.csv")
        mem = read_max_mem(exp_dir / "train.log")
        records.append(
            {
                "experiment": exp_dir.name,
                "final_loss": "" if loss is None else f"{loss:.6f}",
                "max_mem_gb": "" if mem is None else f"{mem:.2f}",
                "loss_curve": str((exp_dir / "train_loss_curve.png").as_posix()),
                "pred_vis": str((exp_dir / "prediction_sample.png").as_posix()),
            }
        )

    summary_csv = root / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["experiment", "final_loss", "max_mem_gb", "loss_curve", "pred_vis"],
        )
        writer.writeheader()
        writer.writerows(records)

    summary_md = root / "summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# SCM Grid Summary\n\n")
        f.write("| experiment | final_loss | max_mem_gb | loss_curve | pred_vis |\n")
        f.write("|---|---:|---:|---|---|\n")
        for r in records:
            f.write(
                f"| {r['experiment']} | {r['final_loss']} | {r['max_mem_gb']} | "
                f"{r['loss_curve']} | {r['pred_vis']} |\n"
            )

    print(f"[Done] summary_csv={summary_csv}")
    print(f"[Done] summary_md={summary_md}")


if __name__ == "__main__":
    main()
