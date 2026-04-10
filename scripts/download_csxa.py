import argparse
import hashlib
import shutil
import time
from pathlib import Path

import requests


CSXA_FILES = {
    "xlsx": {
        "filename": "datasets.xlsx",
        "url": "https://china.scidb.cn/download?fileId=4deb4690ce87b3f491be184fb8d3b41f",
        "size": 8_637_658,
        "md5": "9c6d27010efc62554ffb1f16dc0a8cb3",
    },
    "json": {
        "filename": "datasets-JSON.zip",
        "url": "https://china.scidb.cn/download?fileId=5dada884dd8d622531e826f2452e35d7",
        "size": 1_466_370_159,
        "md5": "bf2e959df985aa5e676302ce92a7ec3c",
    },
    "png": {
        "filename": "datasets-PNG.zip",
        "url": "https://china.scidb.cn/download?fileId=801011b2c734ad280b9326a29358730f",
        "size": 1_394_632_623,
        "md5": "008db86e56f05003fa1870430c245bfd",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download CSXA cervical spine X-ray files from Science Data Bank.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "xray" / "csxa" / "original",
        help="Destination directory for downloaded files.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=["xlsx", "json", "png", "all"],
        default=["xlsx"],
        help='Which CSXA files to download. Default: "xlsx".',
    )
    parser.add_argument(
        "--overwrite",
        "-r",
        action="store_true",
        default=False,
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        default=False,
        help="Skip MD5 verification after download.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.0,
        help="Minimum free disk space buffer to keep after the planned downloads.",
    )
    return parser.parse_args()


def expand_include(values: list[str]) -> list[str]:
    if "all" in values:
        return ["xlsx", "json", "png"]
    return values


def compute_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def check_free_space(output_dir: Path, planned_size: int, min_free_gb: float) -> tuple[int, int]:
    usage = shutil.disk_usage(output_dir.resolve())
    min_free_bytes = int(min_free_gb * (1024**3))
    remaining = usage.free - planned_size
    return usage.free, remaining - min_free_bytes


def stream_download(url: str, destination: Path, expected_size: int, max_attempts: int = 10) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        existing_size = destination.stat().st_size if destination.exists() else 0
        if existing_size == expected_size:
            return
        if existing_size > expected_size:
            destination.unlink()
            existing_size = 0

        headers = {}
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as response:
                response.raise_for_status()
                if existing_size > 0 and response.status_code != 206:
                    destination.unlink(missing_ok=True)
                    existing_size = 0
                    raise RuntimeError("Server did not honor byte-range resume request.")

                mode = "ab" if existing_size > 0 else "wb"
                with destination.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        except (requests.RequestException, RuntimeError) as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise RuntimeError(f"Failed downloading {destination.name} after {attempt} attempts.") from exc
            time.sleep(min(5 * attempt, 30))
            continue

        final_size = destination.stat().st_size if destination.exists() else 0
        if final_size == expected_size:
            return

        attempt += 1
        if attempt >= max_attempts:
            raise RuntimeError(
                f"Download finished with incorrect size for {destination.name}: "
                f"expected {expected_size}, got {final_size}."
            )
        time.sleep(min(5 * attempt, 30))


def main():
    args = parse_args()
    requested = expand_include(args.include)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    planned_downloads = []
    planned_size = 0
    for key in requested:
        entry = CSXA_FILES[key]
        destination = args.output_dir / entry["filename"]
        expected_size = int(entry["size"])
        existing_size = destination.stat().st_size if destination.exists() else 0
        if destination.exists() and not args.overwrite and existing_size == expected_size:
            continue
        planned_downloads.append((key, entry, destination))
        planned_size += max(expected_size - existing_size, 0)

    free_bytes, safe_remaining = check_free_space(args.output_dir, planned_size, args.min_free_gb)
    if safe_remaining < 0:
        raise RuntimeError(
            "Not enough free space for the requested download set. "
            f"Free: {free_bytes / (1024**3):.2f} GB, "
            f"planned: {planned_size / (1024**3):.2f} GB, "
            f"required post-download buffer: {args.min_free_gb:.2f} GB."
        )

    if not planned_downloads:
        print("All requested CSXA files already exist. Nothing to do.")
        return

    for key, entry, destination in planned_downloads:
        print(f'Downloading {key}: {entry["filename"]}')
        stream_download(entry["url"], destination, expected_size=int(entry["size"]))
        actual_size = destination.stat().st_size
        expected_size = int(entry["size"])
        if actual_size != expected_size:
            raise RuntimeError(
                f'File size mismatch for {destination.name}: expected {expected_size}, got {actual_size}.'
            )
        if not args.skip_md5:
            actual_md5 = compute_md5(destination)
            if actual_md5.lower() != str(entry["md5"]).lower():
                raise RuntimeError(
                    f'MD5 mismatch for {destination.name}: expected {entry["md5"]}, got {actual_md5}.'
                )
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
