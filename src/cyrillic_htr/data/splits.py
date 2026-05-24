from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def read_tsv(
    path: str | Path,
    image_column: str,
    text_column: str,
    has_header: bool,
) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"TSV file not found: {path}")

    if has_header:
        dataframe = pd.read_csv(path, sep="\t")
    else:
        dataframe = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=[image_column, text_column],
        )

    return dataframe


def save_tsv(dataframe: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, sep="\t", index=False)


def make_train_val_test_splits(
    train_tsv: str | Path,
    test_tsv: str | Path,
    train_split_tsv: str | Path,
    val_split_tsv: str | Path,
    test_split_tsv: str | Path,
    image_column: str,
    text_column: str,
    tsv_has_header: bool,
    val_size: float,
    random_seed: int,
) -> None:
    train_dataframe = read_tsv(
        train_tsv,
        image_column=image_column,
        text_column=text_column,
        has_header=tsv_has_header,
    )
    test_dataframe = read_tsv(
        test_tsv,
        image_column=image_column,
        text_column=text_column,
        has_header=tsv_has_header,
    )

    train_split, val_split = train_test_split(
        train_dataframe,
        test_size=val_size,
        random_state=random_seed,
        shuffle=True,
    )

    train_paths = set(train_split[image_column].astype(str))
    val_paths = set(val_split[image_column].astype(str))
    test_paths = set(test_dataframe[image_column].astype(str))

    if train_paths & val_paths:
        raise ValueError("Train and validation splits overlap.")
    if train_paths & test_paths:
        raise ValueError("Train and test splits overlap.")
    if val_paths & test_paths:
        raise ValueError("Validation and test splits overlap.")

    save_tsv(train_split, train_split_tsv)
    save_tsv(val_split, val_split_tsv)
    save_tsv(test_dataframe, test_split_tsv)

    print(f"Train split saved to: {train_split_tsv}")
    print(f"Validation split saved to: {val_split_tsv}")
    print(f"Test split saved to: {test_split_tsv}")
    print(f"Train size: {len(train_split)}")
    print(f"Validation size: {len(val_split)}")
    print(f"Test size: {len(test_dataframe)}")
