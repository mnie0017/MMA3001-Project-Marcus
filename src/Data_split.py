"""
Data_split.py
=============

Utility for preparing a single sensor's data for supervised regression:
filters to one sensor, keeps only the relevant columns, coerces them to
numeric, drops incomplete rows, and performs a random train/validation
split.
"""

import numpy as np
import pandas as pd


def split_sensor_data(
    df,
    sensor_id,
    features,
    target="Formaldehyde (µg/m3)",
    sensor_column="sensorid",
    validation_fraction=0.2,
    random_state=42
):
    """Filter a dataset to one sensor and split it into train/validation sets.

    Rows for the given sensor are restricted to the requested feature
    and target columns, coerced to numeric, and any row with a missing
    or infinite value in those columns is dropped. The remaining
    complete rows are then randomly split into training and validation
    subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset containing (at least) the sensor identifier
        column, the feature columns, and the target column.
    sensor_id : Any
        The value in ``sensor_column`` identifying which sensor's rows
        to keep.
    features : list of str
        Names of the predictor columns to use as model inputs.
    target : str, default "Formaldehyde (µg/m3)"
        Name of the column to be predicted.
    sensor_column : str, default "sensorid"
        Name of the column used to filter rows down to one sensor.
    validation_fraction : float, default 0.2
        Fraction of the complete, filtered observations to set aside
        as the validation set (rounded to the nearest whole number of
        rows, with at least 1 row always reserved for validation).
    random_state : int, default 42
        Seed for the random number generator used to select which
        rows go into the validation set, for reproducibility.

    Returns
    -------
    train_df : pandas.DataFrame
        The training subset (features + target columns only).
    val_df : pandas.DataFrame
        The validation subset (features + target columns only).

    Raises
    ------
    ValueError
        If fewer than 2 complete observations remain after filtering
        and cleaning, or if the computed validation set size would be
        greater than or equal to the total number of available rows.
    """

    # Select the specified sensor
    # (Keep only the rows belonging to the requested sensor_id.)
    data = df.loc[df[sensor_column] == sensor_id].copy()

    # Select predictors and target
    # (Discard every other column; we only need features + target
    # for training/validation.)
    columns = features + [target]
    data = data[columns].copy()

    # Ensure regression variables are numeric
    # (Any value that can't be converted to a number becomes NaN,
    # e.g. stray strings or malformed entries.)
    data = data.apply(pd.to_numeric, errors="coerce")

    # Exclude incomplete observations from model training
    # (Treat +/- infinity as missing too, then drop any row with a
    # NaN in any of the feature or target columns.)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=columns)

    # Check there are enough complete observations
    # (Need at least 2 rows to form a non-trivial train/validation split.)
    if len(data) < 2:
        raise ValueError(
            "Not enough complete observations to split."
        )

    # Randomly split into training and validation sets
    # (Use a seeded generator so the split is reproducible across runs.)
    rng = np.random.default_rng(random_state)

    # Number of rows to allocate to validation: round to nearest whole
    # row count, but always reserve at least 1 row.
    n_validation = max(
        1, int(round(len(data) * validation_fraction))
    )

    # Guard against a validation_fraction so large (or dataset so
    # small) that no rows would be left for training.
    if n_validation >= len(data):
        raise ValueError(
            "Not enough observations for the requested split."
        )

    # Randomly choose n_validation row positions (without replacement)
    # to assign to the validation set.
    validation_positions = rng.choice(
        len(data), size=n_validation, replace=False
    )

    # Build a boolean mask marking which rows (by position) are
    # validation rows, so the remainder can be used as training rows.
    validation_mask = np.zeros(len(data), dtype=bool)
    validation_mask[validation_positions] = True

    # Rows NOT selected for validation become the training set;
    # the selected rows become the validation set.
    train_df = data.iloc[~validation_mask].copy()
    val_df = data.iloc[validation_mask].copy()

    return train_df, val_df
