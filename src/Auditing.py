"""
Auditing.py
===========

Loads the raw environmental sensor CSV, unpacks the nested JSON payload
in each row into flat columns, removes duplicate records, and runs a
basic physical-limits audit (min/max/negative-value counts) on the
key measurement columns.

Running this file directly (``python Auditing.py``) loads and cleans
the default dataset and prints a preview of the result.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display


# Resolve the project root (two levels up from this file: src/ -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default location of the raw sensor data CSV, relative to the project root.
DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "5EnvSensor_MayToDec2024_180kRows.csv"
)


def parse_all_env_data(df_chunk):
    """Extract environmental variables from each row's nested JSON.

    Each row of the raw CSV has a ``jsondata`` column containing a
    (possibly JSON-encoded string) list of variable readings, e.g.
    temperature, humidity, CO2, etc. This function unpacks that list
    into a single flat dictionary per row, where each variable becomes
    its own column (named ``"<variable name> (<unit>)"`` when a unit
    is present), alongside a handful of metadata columns carried over
    directly from the row.

    Parameters
    ----------
    df_chunk : pandas.DataFrame
        A chunk of the raw CSV, expected to contain at least the
        columns ``id``, ``sensorid``, ``devicetype``, ``status``,
        ``createdate``, ``processdate``, and ``jsondata``.

    Returns
    -------
    pandas.DataFrame
        One row per successfully parsed input row, with metadata
        columns plus one column per extracted variable. Rows whose
        JSON could not be parsed are silently skipped.
    """
    records = []

    for _, row in df_chunk.iterrows():
        try:
            raw_json = row["jsondata"]

            # jsondata may already be a parsed Python list, or it may
            # be a JSON-encoded string that needs decoding first.
            if isinstance(raw_json, str):
                json_list = json.loads(raw_json)
            else:
                json_list = raw_json

            # Skip rows whose jsondata isn't a list of readings
            # (e.g. malformed or unexpected structure).
            if not isinstance(json_list, list):
                continue

            # Start the flattened record with the row's metadata
            # columns, carried over as-is.
            record = {
                "id": row.get("id"),
                "sensorid": row.get("sensorid"),
                "devicetype": row.get("devicetype"),
                "status": row.get("status"),
                "createdate": row.get("createdate"),
                "processdate": row.get("processdate"),
            }

            # Each element of json_list represents one measured
            # variable (e.g. temperature) with its own name, unit,
            # and list of values; take the first value for each.
            for element in json_list:
                var_info = element.get("variable", {})
                var_name = var_info.get("name")
                var_unit = var_info.get("unit", "")
                values = element.get("values", [])

                if values and var_name:
                    value = values[0].get("value")
                    # Build a column name that includes the unit when
                    # one is available, e.g. "Temperature (°C)".
                    col_name = (
                        f"{var_name} ({var_unit})"
                        if var_unit
                        else var_name
                    )
                    record[col_name] = value

            records.append(record)

        except (ValueError, TypeError, KeyError, AttributeError):
            # Skip records whose JSON cannot be parsed.
            continue

    return pd.DataFrame(records)


def load_and_clean_data(
    env_path=DEFAULT_DATA_PATH,
    chunksize=30000,
):
    """Load, extract, deduplicate and audit the sensor dataset.

    Reads the raw CSV in chunks (to limit memory usage on large
    files), flattens each row's nested JSON via
    :func:`parse_all_env_data`, concatenates the results, removes
    duplicate records, and prints a physical-limits audit (min, max,
    and count of negative values) for each key measurement column.

    Parameters
    ----------
    env_path : str or pathlib.Path, default DEFAULT_DATA_PATH
        Path to the raw sensor CSV file.
    chunksize : int, default 30000
        Number of rows to read from the CSV at a time.

    Returns
    -------
    pandas.DataFrame
        The flattened, deduplicated dataset, with the physical
        measurement columns coerced to numeric.

    Raises
    ------
    FileNotFoundError
        If no file exists at ``env_path``.
    ValueError
        If no data could be extracted from the CSV at all (e.g. every
        chunk failed to parse or the file was empty).
    """
    env_path = Path(env_path)

    if not env_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {env_path}\n"
            "Place the CSV in the repository's data/ directory."
        )

    chunks = []

    # Read and parse the CSV chunk by chunk, rather than all at once,
    # to keep memory usage manageable for large files.
    for chunk in pd.read_csv(env_path, chunksize=chunksize):
        chunks.append(parse_all_env_data(chunk))

    if not chunks:
        raise ValueError("No data was extracted from the CSV.")

    # Combine all parsed chunks into one DataFrame.
    cleaned_df = pd.concat(chunks, ignore_index=True)

    original_count = len(cleaned_df)

    # Remove duplicate readings: same sensor reporting the same
    # createdate more than once is treated as a duplicate record.
    cleaned_df = cleaned_df.drop_duplicates(
        subset=["sensorid", "createdate"]
    ).reset_index(drop=True)

    print(f"Removed {original_count - len(cleaned_df)} duplicates.")
    print(f"Total records after cleaning: {len(cleaned_df)}")

    # Columns expected to hold physically meaningful numeric
    # measurements, used for the sanity-check audit below.
    physical_cols = [
        "Carbon dioxide (ppm)",
        "Temperature (°C)",
        "Humidity (HR%)",
        "Pressure (mb)",
        "Formaldehyde (µg/m3)",
        "Light Volatile Organic Compounds (ppb)",
        "Particulate matter 1 (µg/m3)",
        "Particulate matter 2.5 (µg/m3)",
    ]

    print("\n=== PHYSICAL LIMITS AUDIT ===")

    # For each expected measurement column (if present), coerce to
    # numeric and report its min, max, and how many negative values
    # it contains, negative readings are physically implausible for
    # these variables and may indicate a data quality issue.
    for col in physical_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(
                cleaned_df[col], errors="coerce"
            )

            neg_count = int((cleaned_df[col] < 0).sum())
            min_val = cleaned_df[col].min()
            max_val = cleaned_df[col].max()

            print(
                f"{col: <40} | Min: {min_val: >8} "
                f"| Max: {max_val: >8} "
                f"| Negative Count: {neg_count}"
            )

    return cleaned_df


if __name__ == "__main__":
    # When run as a script, load/clean the default dataset and show
    # a quick preview of the resulting DataFrame.
    cleaned_df = load_and_clean_data()
    print("\nExtracted data preview:")
    display(cleaned_df.head())
    cleaned_df.info()
