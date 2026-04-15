import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


CTW1500_TEST_IMAGES_URL = "https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip"
CTW1500_TEST_LABELS_URL = "https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and not force:
        print(f"[Skip] Exists: {dst}")
        return
    ensure_dir(dst.parent)
    print(f"[Download] {url}")
    urllib.request.urlretrieve(url, dst)
    print(f"[Saved] {dst}")


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_contents(src: Path, dst: Path) -> None:
    ensure_dir(dst)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def pick_source_root(extract_root: Path, preferred_names: list[str]) -> Path:
    for name in preferred_names:
        candidate = extract_root / name
        if candidate.exists():
            return candidate

    top_level = list(extract_root.iterdir())
    if len(top_level) == 1 and top_level[0].is_dir():
        return top_level[0]
    return extract_root


def extract_zip_to_target(
    zip_path: Path,
    target_dir: Path,
    preferred_source_names: list[str],
    force: bool = False,
) -> None:
    if force:
        clear_dir(target_dir)
    else:
        ensure_dir(target_dir)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_path)
        src_root = pick_source_root(tmp_path, preferred_source_names)
        copy_tree_contents(src_root, target_dir)

    print(f"[Extracted] {zip_path.name} -> {target_dir}")


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CTW1500 test split.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    raw_root = root / "data" / "raw" / "ctw1500"
    downloads_dir = raw_root / "downloads"
    imgs_test_dir = raw_root / "imgs" / "test"
    ann_test_dir = raw_root / "annotations" / "test"
    processed_dir = root / "data" / "processed" / "ctw1500"

    for path in [downloads_dir, imgs_test_dir, ann_test_dir, processed_dir, root / "docs"]:
        ensure_dir(path)

    images_zip = downloads_dir / "ctw1500_test_images.zip"
    labels_zip = downloads_dir / "ctw1500_test_labels.zip"

    if not args.skip_download:
        download_file(CTW1500_TEST_IMAGES_URL, images_zip, force=args.force)
        download_file(CTW1500_TEST_LABELS_URL, labels_zip, force=args.force)

    if not images_zip.exists() or not labels_zip.exists():
        raise FileNotFoundError("Zip files not found. Remove --skip-download or place zip files into downloads dir.")

    extract_zip_to_target(
        images_zip,
        imgs_test_dir,
        preferred_source_names=["test_images", "test", "imgs"],
        force=args.force,
    )
    extract_zip_to_target(
        labels_zip,
        ann_test_dir,
        preferred_source_names=["test", "ctw1500_test_labels", "labels"],
        force=args.force,
    )

    print("[Done] CTW1500 test split prepared.")
    print(f"[Count] images/test files: {count_files(imgs_test_dir)}")
    print(f"[Count] annotations/test files: {count_files(ann_test_dir)}")
    print(f"[Path] root: {root}")


if __name__ == "__main__":
    main()
