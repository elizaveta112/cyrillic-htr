from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

STEP_METRICS = {
    "train_loss_step",
    "lr-AdamW",
}

EPOCH_METRICS = {
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
}


def get_x_column(metric_name: str) -> str:
    if metric_name in STEP_METRICS:
        return "step"

    return "epoch"


def read_metrics_csv(metrics_csv_path: str | Path) -> pd.DataFrame:
    metrics_csv_path = Path(metrics_csv_path)

    if not metrics_csv_path.exists():
        return pd.DataFrame()

    dataframe = pd.read_csv(metrics_csv_path)
    dataframe["source_file"] = str(metrics_csv_path)

    return dataframe


def read_all_metrics(metrics_csv_path: str | Path) -> pd.DataFrame:
    metrics_csv_path = Path(metrics_csv_path)
    metrics_files = sorted(metrics_csv_path.parent.parent.glob("version_*/metrics.csv"))

    if not metrics_files:
        metrics_files = [metrics_csv_path]

    dataframes = [read_metrics_csv(path) for path in metrics_files]
    dataframes = [dataframe for dataframe in dataframes if not dataframe.empty]

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(dataframes, ignore_index=True, sort=False)


def plot_metric(
    metrics_dataframe: pd.DataFrame,
    metric_name: str,
    output_path: str | Path,
) -> None:
    if metric_name not in metrics_dataframe.columns:
        return

    x_column = get_x_column(metric_name=metric_name)

    if x_column not in metrics_dataframe.columns:
        return

    metric_dataframe = metrics_dataframe[[x_column, metric_name]].dropna()

    if metric_dataframe.empty:
        return

    metric_dataframe = metric_dataframe.sort_values(x_column)
    metric_dataframe = metric_dataframe.drop_duplicates(
        subset=[x_column],
        keep="last",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(
        metric_dataframe[x_column],
        metric_dataframe[metric_name],
        marker="o",
        markersize=3,
    )
    plt.xlabel(x_column)
    plt.ylabel(metric_name)
    plt.title(metric_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_metrics(
    metrics_csv_path: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    metrics_dataframe = read_all_metrics(metrics_csv_path)

    if metrics_dataframe.empty:
        print(f"No metrics found, skipping plots: {metrics_csv_path}")
        return

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
