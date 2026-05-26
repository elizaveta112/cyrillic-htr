from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(
    metrics_dataframe: pd.DataFrame,
    metric_name: str,
    output_path: str | Path,
) -> None:
    if metric_name not in metrics_dataframe.columns:
        return

    metric_dataframe = metrics_dataframe[["step", metric_name]].dropna()

    if metric_dataframe.empty:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(metric_dataframe["step"], metric_dataframe[metric_name])
    plt.xlabel("step")
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_metrics(
    metrics_csv_path: str | Path,
    output_dir: str | Path,
) -> None:
    metrics_csv_path = Path(metrics_csv_path)
    output_dir = Path(output_dir)

    if not metrics_csv_path.exists():
        print(f"Metrics CSV not found, skipping plots: {metrics_csv_path}")
        return

    metrics_dataframe = pd.read_csv(metrics_csv_path)

    metric_names = [
        "train_loss_step",
        "train_loss_epoch",
        "val_loss",
        "val_cer",
        "val_wer",
        "val_line_accuracy",
        "val_edit_similarity",
        "val_valid_character_rate",
        "test_loss",
        "test_cer",
        "test_wer",
        "test_line_accuracy",
        "test_edit_similarity",
        "test_valid_character_rate",
    ]

    for metric_name in metric_names:
        plot_metric(
            metrics_dataframe=metrics_dataframe,
            metric_name=metric_name,
            output_path=output_dir / f"{metric_name}.png",
        )

    print(f"Training plots saved to: {output_dir}")
