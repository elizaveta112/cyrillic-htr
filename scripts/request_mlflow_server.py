import base64
import json
import urllib.request
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf


def encode_image_base64(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def build_invocations_url(config: DictConfig) -> str:
    host = str(config.serving.host)
    port = int(config.serving.port)
    return f"http://{host}:{port}/invocations"


def build_request_payload(config: DictConfig) -> dict[str, Any]:
    image_path = OmegaConf.select(config, "infer.image_path", default=None)

    if image_path is None:
        raise ValueError("Provide image path with +infer.image_path=path/to/image.png")

    image_path = Path(str(image_path))

    return {
        "dataframe_records": [
            {
                "image_path": str(image_path),
                "image_base64": encode_image_base64(image_path),
            }
        ]
    }


def send_request(config: DictConfig) -> dict[str, Any]:
    url = build_invocations_url(config)
    payload = build_request_payload(config)

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    timeout_seconds = float(config.serving.request_timeout_seconds)

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_payload = response.read().decode("utf-8")

    return json.loads(response_payload)


def save_response(config: DictConfig, response: dict[str, Any]) -> None:
    output_path = Path(str(config.serving.response_output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"MLflow serving response saved to: {output_path}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    response = send_request(config)
    save_response(config, response)
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
