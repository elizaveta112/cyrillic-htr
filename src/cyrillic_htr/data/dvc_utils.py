import subprocess
from pathlib import Path


def dvc_pull(target: str | Path | None = None, remote: str | None = None) -> None:
    command = ["dvc", "pull"]

    if target is not None:
        command.append(str(target))

    if remote is not None:
        command.extend(["-r", remote])

    subprocess.run(command, check=True)
