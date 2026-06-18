"""Download Lora (TTF) and Latin Modern (OTF) fonts for scikit-rom.

Run from the repository root:

    python scripts/download_skrom_fonts.py

or run this file from anywhere with:

    python download_skrom_fonts.py --root /path/to/scikit-rom

The files are downloaded from official upstream sources:
- Lora: Google Fonts repository
- Latin Modern: CTAN mirror
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


LORA_FILES = {
    "Lora[wght].ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/Lora%5Bwght%5D.ttf",
    "Lora-Italic[wght].ttf": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/Lora-Italic%5Bwght%5D.ttf",
    "OFL.txt": "https://raw.githubusercontent.com/google/fonts/main/ofl/lora/OFL.txt",
}

LATIN_MODERN_ZIP = "https://mirrors.ctan.org/fonts/lm.zip"
LATIN_MODERN_MATH_ZIP = "https://mirrors.ctan.org/fonts/lm-math.zip"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"download: {url}")
    urllib.request.urlretrieve(url, destination)


def download_lora(font_root: Path) -> None:
    lora_dir = font_root / "lora"
    lora_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in LORA_FILES.items():
        download(url, lora_dir / filename)


def extract_matching(zip_path: Path, destination: Path, suffixes: tuple[str, ...]) -> int:
    count = 0
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.namelist():
            lower = member.lower()
            if not lower.endswith(suffixes):
                continue
            if "/doc/" in lower or lower.startswith("doc/"):
                continue
            target = destination / Path(member).name
            if not target.name:
                continue
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    return count


def download_latin_modern(font_root: Path, work_dir: Path) -> None:
    lm_dir = font_root / "latinmodern"
    lm_dir.mkdir(parents=True, exist_ok=True)

    lm_zip = work_dir / "lm.zip"
    download(LATIN_MODERN_ZIP, lm_zip)
    n_fonts = extract_matching(lm_zip, lm_dir, (".otf", ".ttf"))
    print(f"extracted {n_fonts} Latin Modern text font files")

    lm_math_zip = work_dir / "lm-math.zip"
    try:
        download(LATIN_MODERN_MATH_ZIP, lm_math_zip)
        n_math = extract_matching(lm_math_zip, lm_dir, (".otf", ".ttf"))
        print(f"extracted {n_math} Latin Modern Math font files")
    except Exception as exc:
        print(f"warning: could not download Latin Modern Math: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing src/skrom. Defaults to current directory.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    font_root = root / "src" / "skrom" / "assets" / "fonts"
    work_dir = root / ".font_downloads"
    work_dir.mkdir(parents=True, exist_ok=True)

    download_lora(font_root)
    download_latin_modern(font_root, work_dir)

    print("\nDone. Fonts are in:")
    print(font_root)
    print("\nAdd this to pyproject.toml if not already present:\n")
    print('[tool.setuptools.package-data]')
    print('"skrom" = ["assets/fonts/**/*.ttf", "assets/fonts/**/*.otf", "assets/fonts/**/*.txt", "assets/fonts/**/LICENSE*"]')


if __name__ == "__main__":
    main()
