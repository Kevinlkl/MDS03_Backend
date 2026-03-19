import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

def iter_image_files(root_dir: Path) -> Iterable[Path]:
    for path in root_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path

def resize_dataset_images(
    input_dir: Path,
    output_dir: Path,
    target_size: tuple[int, int] = (512, 512)
) -> None:
    source_root = Path(input_dir)
    output_root = Path(output_dir)

    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist or is not a directory.")
    
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_files(source_root))

    total = len(image_paths)
    success = 0
    skipped = 0

    for src_path in tqdm(image_paths, desc="Resizing images"):
        relative_path = src_path.relative_to(source_root)

        #change extension to .png
        dst_path = (output_root / relative_path).with_suffix('.png')
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                resized = img.resize((target_size[0], target_size[1]), Image.Resampling.LANCZOS)
                resized.save(dst_path, format='PNG')
                success += 1
        except (UnidentifiedImageError, OSError) as exc:
            skipped += 1
            print(f"Skipped {src_path}: {exc}")

    print("\nResize complete")
    print(f"Input directory : {source_root}")
    print(f"Output directory: {output_root}")
    print(f"Target size     : {target_size[0]}x{target_size[1]}")
    print(f"Found images    : {total}")
    print(f"Resized         : {success}")
    print(f"Skipped         : {skipped}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch resize dataset images")

    parser.add_argument(
        "--input",
        default="Dataset/PET",
        help="Input dataset directory",
    )

    parser.add_argument(
        "--output",
        default="Dataset/PET_256",
        help="Output directory for resized images",
    )

    parser.add_argument(
        "--width",
        default=256,
        type=int,
        help="Target width for resizing",
    )
    parser.add_argument(
        "--height",
        default=256,
        type=int,
        help="Target height for resizing",
    )

    return parser.parse_args()

def print_first_n_metadata(folder: Path, n: int = 10) -> None:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        print(f"Metadata folder not found: {folder}")
        return

    files = sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    )[:n]

    print(f"\nMetadata for first {len(files)} images in {folder}:")
    for i, p in enumerate(files, 1):
        try:
            stat = p.stat()
            with Image.open(p) as img:
                modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{i}] {p.name}")
                print(f"  format   : {img.format}")
                print(f"  mode     : {img.mode}")
                print(f"  size     : {img.size[0]}x{img.size[1]}")
                print(f"  file size: {stat.st_size} bytes")
                print(f"  modified : {modified}")
                print("-" * 50)
        except (UnidentifiedImageError, OSError) as exc:
            print(f"[{i}] {p.name} -> cannot read metadata: {exc}")


if __name__ == "__main__":
    args = parse_args()
    resize_dataset_images(args.input, args.output, (args.width, args.height))
    print_first_n_metadata(Path("Dataset/Pituitary_256_T1weighted"), n=10)