from pathlib import Path

import pandas as pd

from cyrillic_htr.training.plots import plot_training_metrics


def test_plot_training_metrics(tmp_path: Path) -> None:
    metrics_csv_path = tmp_path / "metrics.csv"
    output_dir = tmp_path / "plots"

    dataframe = pd.DataFrame(
        {
            "step": [0, 1, 2],
            "epoch": [0, 0, 0],
            "train_loss_step": [3.0, 2.0, 1.0],
            "val_cer": [None, None, 0.6],
            "val_wer": [None, None, 0.7],
        }
    )
    dataframe.to_csv(metrics_csv_path, index=False)

    plot_training_metrics(
        metrics_csv_path=metrics_csv_path,
        output_dir=output_dir,
    )

    assert (output_dir / "train_loss_step.png").exists()
    assert (output_dir / "val_cer.png").exists()
    assert (output_dir / "val_wer.png").exists()


def test_plot_training_metrics_with_single_point(tmp_path: Path) -> None:
    metrics_csv_path = tmp_path / "metrics.csv"
    output_dir = tmp_path / "plots"

    dataframe = pd.DataFrame(
        {
            "step": [1],
            "epoch": [0],
            "val_cer": [1.0],
        }
    )
    dataframe.to_csv(metrics_csv_path, index=False)

    plot_training_metrics(
        metrics_csv_path=metrics_csv_path,
        output_dir=output_dir,
    )

    assert (output_dir / "val_cer.png").exists()
