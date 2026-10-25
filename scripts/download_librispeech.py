"""Download and extract LibriSpeech dev-clean dataset.

Source: OpenSLR Resource 12 (LibriSpeech ASR corpus)
Subset: dev-clean (~337 MB compressed, 40 speakers, ~5.4 hours)
"""

import sys
import tarfile
import urllib.request
from pathlib import Path


URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
DATA_DIR = Path("data")
ARCHIVE_PATH = DATA_DIR / "dev-clean.tar.gz"
EXTRACT_DIR = DATA_DIR / "LibriSpeech" / "dev-clean"


def download_with_progress(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {url}")
    print(f"Saving to:       {output_path.resolve()}")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent:5.1f}% | {mb_downloaded:6.1f} MB / {mb_total:6.1f} MB")
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            sys.stdout.write(f"\rDownloaded: {mb_downloaded:6.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(output_path), reporthook=reporthook)
    print("\nDownload complete.")


def extract_archive(archive_path: Path, target_dir: Path):
    print(f"Extracting {archive_path.name}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=target_dir.parent.parent)
    print(f"Extraction complete: {target_dir.resolve()}")


def verify_dataset(dataset_dir: Path):
    flac_files = list(dataset_dir.rglob("*.flac"))
    speakers = set(p.parent.parent.name for p in flac_files)
    print(f"\nDataset Verification:")
    print(f"  • Total .flac files: {len(flac_files)}")
    print(f"  • Unique speakers:   {len(speakers)}")
    assert len(flac_files) > 1000, f"Expected >1000 files, found {len(flac_files)}"
    assert len(speakers) >= 30, f"Expected >=30 speakers, found {len(speakers)}"
    print("Verification passed successfully.")


def main():
    if EXTRACT_DIR.exists() and len(list(EXTRACT_DIR.rglob("*.flac"))) > 1000:
        print(f"Dataset already exists at: {EXTRACT_DIR.resolve()}")
        verify_dataset(EXTRACT_DIR)
        return

    if not ARCHIVE_PATH.exists() or ARCHIVE_PATH.stat().st_size < 300 * 1024 * 1024:
        download_with_progress(URL, ARCHIVE_PATH)
    else:
        print(f"Found existing archive ({ARCHIVE_PATH.stat().st_size / (1024*1024):.1f} MB), skipping download.")

    extract_archive(ARCHIVE_PATH, EXTRACT_DIR)
    verify_dataset(EXTRACT_DIR)

    # Clean up archive to save 337 MB disk space
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
        print(f"Cleaned up archive: {ARCHIVE_PATH.name}")


if __name__ == "__main__":
    main()
