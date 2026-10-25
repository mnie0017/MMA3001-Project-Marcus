"""
test_project.py
===============

Contains unit and integration tests using pytest to verify core functionality
of the environmental sensor cleaning, splitting, and neural network pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.Auditing import parse_all_env_data
from src.Data_split import split_sensor_data


def test_parse_all_env_data():
    """Verify that nested JSON payloads are parsed into flat columns correctly."""
    # Prepare a mock record representing one raw data row
    mock_row = {
        "id": 1,
        "sensorid": 9999,
        "devicetype": "env",
        "status": "Processed",
        "createdate": "2024-06-28 01:00:00+10",
        "processdate": "2024-06-28 01:01:00+10",
        "jsondata": [
            {
                "variable": {"name": "Temperature", "unit": "°C"},
                "values": [{"value": 22.5}]
            },
            {
                "variable": {"name": "Humidity", "unit": "HR%"},
                "values": [{"value": 50.0}]
            }
        ]
    }
    
    df_mock = pd.DataFrame([mock_row])
    parsed_df = parse_all_env_data(df_mock)
    
    assert len(parsed_df) == 1
    assert "Temperature (°C)" in parsed_df.columns
    assert "Humidity (HR%)" in parsed_df.columns
    assert parsed_df.loc[0, "Temperature (°C)"] == 22.5
    assert parsed_df.loc[0, "Humidity (HR%)"] == 50.0


def test_split_sensor_data():
    """Verify dataset partitioning behaves deterministically and filters NaN/Inf."""
    # Create dummy DataFrame with clean values, NaNs, and Infs
    data = {
        "sensorid": [100, 100, 100, 100, 100, 100, 200],
        "Temperature (°C)": [20.0, 21.0, 22.0, 23.0, np.nan, 25.0, 30.0],
        "Formaldehyde (µg/m3)": [5.0, 6.0, 7.0, 8.0, 9.0, np.inf, 12.0]
    }
    df = pd.DataFrame(data)
    
    # Splitting configuration
    features = ["Temperature (°C)"]
    target = "Formaldehyde (µg/m3)"
    
    train_df, val_df = split_sensor_data(
        df=df,
        sensor_id=100,
        features=features,
        target=target,
        sensor_column="sensorid",
        validation_fraction=0.25,
        random_state=42
    )
    
    # Out of 6 rows for sensor 100:
    # Row 4 has NaN temperature -> dropped
    # Row 5 has Inf target -> dropped
    # Remaining complete rows = 4. Validation fraction 0.25 -> 1 validation row, 3 training rows.
    assert len(train_df) == 3
    assert len(val_df) == 1
    
    # Assert no nulls or inf values exist in the splits
    assert not train_df.isna().any().any()
    assert not val_df.isna().any().any()
    assert np.isfinite(train_df.values).all()
