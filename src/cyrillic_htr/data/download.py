import subprocess
from pathlib import Path


def download_data(
    dataset_name: str,
    output_dir: str | Path,
    force_download: bool = False,
) -> None:
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()) and not force_download:
        print(f"Dataset already exists: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_name,
        "-p",
        str(output_dir),
        "--unzip",
    ]

    subprocess.run(command, check=True)
    print(f"Dataset downloaded to: {output_dir}")
