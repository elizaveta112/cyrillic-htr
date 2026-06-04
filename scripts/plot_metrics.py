from pathlib import Path

import hydra
from omegaconf import DictConfig

from cyrillic_htr.training.plots import plot_training_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    log_dir = Path(config.train.log_dir)
    metrics_files = sorted(log_dir.glob("csv/version_*/metrics.csv"))

    if not metrics_files:
        raise FileNotFoundError(f"No metrics.csv files found in: {log_dir}")

    latest_metrics_file = metrics_files[-1]

    plot_training_metrics(
        metrics_csv_path=latest_metrics_file,
        output_dir=config.train.plots_dir,
    )


if __name__ == "__main__":
    main()
