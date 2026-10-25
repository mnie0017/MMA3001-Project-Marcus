"""
NeuralNetwork.py
=================

Trains a neural network regressor to predict
Formaldehyde (HCHO) concentration from a set of sensor-measured features.

Workflow performed by this script:
1. Load and clean the raw sensor dataset.
2. Split the data for one specific sensor into training and validation sets.
3. Build and train a scaled neural-network regression pipeline.
4. Evaluate the trained model on the validation set (MAE, RMSE, R², bias).
5. Perform a residual analysis and save the largest errors for inspection.
6. Generate two diagnostic plots: predicted-vs-actual and residuals-vs-predicted.
7. Re-fit a final model on all available labelled data for the sensor and use
   it to reconstruct (impute) any missing Formaldehyde readings that have
   complete feature data.

All intermediate results (metrics, residuals, plots, reconstructed values)
are written to the `results/` directory under the project root.
"""

from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")  # Save plots without requiring a display.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.Auditing import load_and_clean_data
from src.Data_split import split_sensor_data


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Unique identifier of the sensor whose data will be used for
# training, validation, and missing-value reconstruction.
SENSOR_ID = 6012002000869

# Name of the column this model is trained to predict.
TARGET = "Formaldehyde (µg/m3)"

# Columns used as model inputs (predictors).
FEATURES = [
    "Temperature (°C)",
    "Humidity (HR%)",
    "Carbon dioxide (ppm)",
    "Light Volatile Organic Compounds (ppb)",
    "Particulate matter 1 (µg/m3)",
    "Particulate matter 2.5 (µg/m3)",
]

# Resolve the project root (two levels up from this file: src/ -> project root)
# and ensure the results directory exists before anything is written to it.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load and clean data
# ------------------------------------------------------------

# Load the full dataset (all sensors) and apply any cleaning steps
# defined in src/Auditing.py (e.g. dropping invalid rows, fixing types).
cleaned_df = load_and_clean_data()

# ------------------------------------------------------------
# Split labelled data into training and validation sets
# ------------------------------------------------------------

# Restrict to the chosen sensor, keep only rows where the target is
# known (labelled), and split those into train/validation sets.
train_df, val_df = split_sensor_data(
    df=cleaned_df,
    sensor_id=SENSOR_ID,
    features=FEATURES,
    target=TARGET,
    sensor_column="sensorid",
    validation_fraction=0.2,
    random_state=42,  # Fixed seed for a reproducible split.
)

# Separate each split into inputs (X) and target (y).
X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_val = val_df[FEATURES]
y_val = val_df[TARGET]


# ------------------------------------------------------------
# Define the neural network
# ------------------------------------------------------------

def make_model():
    """Create a scaled neural-network regression pipeline.

    The returned pipeline first standardises the input features
    (zero mean, unit variance) via ``StandardScaler``, then feeds
    them into an ``MLPRegressor`` with two hidden layers.

    Standardising inputs is important for MLPs because the network's
    weight updates (and the ``adam`` solver's internal statistics)
    are sensitive to the input feature scale; without scaling,
    features with larger numeric ranges (e.g. CO2 in ppm) could
    dominate training over features with smaller ranges (e.g. PM1).

    Returns
    -------
    sklearn.pipeline.Pipeline
        A fitted-on-demand pipeline of ``StandardScaler`` followed by
        ``MLPRegressor``, ready to be trained with ``.fit(X, y)`` and
        used for inference with ``.predict(X)``.

    Notes
    -----
    Key hyperparameters used here:
        - ``hidden_layer_sizes=(30, 15)``: two hidden layers with
          30 and 15 neurons respectively.
        - ``activation="relu"``: ReLU activation for hidden units.
        - ``solver="adam"``: Adam optimiser for weight updates.
        - ``alpha=0.001``: L2 regularisation strength.
        - ``learning_rate_init=0.01``: initial learning rate for Adam.
        - ``max_iter=2500``: maximum number of training epochs.
        - ``early_stopping=True``: stop training early using a held-out
          slice of the training data (see ``validation_fraction``) if
          validation score stops improving, to reduce overfitting.
        - ``validation_fraction=0.15``: fraction of the *training* data
          (not the outer validation set) reserved for early stopping.
        - ``n_iter_no_change=35``: number of epochs with no improvement
          allowed before stopping early.
        - ``tol=1e-5``: minimum improvement in loss to count as progress.
        - ``random_state=42``: fixed seed for reproducible weight
          initialisation and training behaviour.
    """

    neural_network = MLPRegressor(
        hidden_layer_sizes=(30, 15),
        activation="relu",
        solver="adam",
        alpha=0.001,
        learning_rate_init=0.01,
        max_iter=2500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=35,
        tol=1e-5,
        random_state=42,
    )

    return make_pipeline(
        StandardScaler(),
        neural_network,
    )


# ------------------------------------------------------------
# Train and validate
# ------------------------------------------------------------

# Instantiate a fresh (untrained) copy of the pipeline.
model = make_model()

# Fit the model on the training split only, timing how long it takes.
start_time = perf_counter()
model.fit(X_train, y_train)
training_time = perf_counter() - start_time

# Generate predictions on the held-out validation split, timing inference.
start_time = perf_counter()
y_pred = model.predict(X_val)
prediction_time = perf_counter() - start_time

# Convert both the ground-truth and predicted values to plain numpy
# arrays so downstream metric functions and arithmetic behave consistently.
y_actual = y_val.to_numpy()
y_pred = np.asarray(y_pred)

# Standard regression accuracy metrics computed on the validation set.
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r_squared = r2_score(y_actual, y_pred)

# Mean signed error: tells us whether the model systematically over-
# or under-predicts, as opposed to MAE/RMSE which only measure magnitude.
# Positive bias means predictions are higher than actual values.
bias = float(np.mean(y_pred - y_actual))

# Collect all headline metrics into a single DataFrame for easy printing
# and export, alongside timing information for training/inference.
metrics = pd.DataFrame({
    "Metric": [
        "MAE",
        "RMSE",
        "R²",
        "Bias (prediction - actual)",
        "Training time (s)",
        "Validation prediction time (s)",
    ],
    "Value": [
        mae,
        rmse,
        r_squared,
        bias,
        training_time,
        prediction_time,
    ],
})

print("\n=== VALIDATION RESULTS ===")
print(
    metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

# Plain-language interpretation of the bias sign for quick reading
# in the console output (doesn't affect the saved metrics).
if bias > 0:
    print("\nThe model tends to OVERestimate formaldehyde.")
elif bias < 0:
    print("\nThe model tends to UNDERestimate formaldehyde.")
else:
    print("\nThe model has approximately zero mean bias.")

# Persist the metrics table to disk for later reference/reporting.
metrics.to_csv(RESULTS_DIR / "validation_metrics.csv", index=False)


# ------------------------------------------------------------
# Residual analysis
# ------------------------------------------------------------

# Residual = actual - predicted:
# positive means underprediction; negative means overprediction.

# Build a per-observation table of actual vs predicted values, keeping
# the original validation DataFrame's index so rows can be traced back
# to the source data if needed.
error_df = pd.DataFrame({
    "Actual": y_actual,
    "Predicted": y_pred,
}, index=val_df.index)

# Signed residual (note the sign convention differs from `bias` above:
# here positive = underprediction, whereas `bias` above is
# prediction - actual, i.e. positive = overprediction).
error_df["Residual"] = (
    error_df["Actual"] - error_df["Predicted"]
)
# Magnitude of error, useful for ranking/sorting regardless of direction.
error_df["Absolute error"] = error_df["Residual"].abs()
# Squared error, useful for relating back to RMSE / loss-style metrics.
error_df["Squared error"] = error_df["Residual"] ** 2

print("\n=== RESIDUAL SUMMARY ===")
# Standard descriptive statistics (count, mean, std, min, quartiles, max)
# of the residual distribution.
print(error_df["Residual"].describe().to_string())

print("\n=== LARGEST ABSOLUTE ERRORS ===")
# Inspect the 10 worst individual predictions to spot potential outliers
# or systematic issues with specific conditions.
print(
    error_df.nlargest(10, "Absolute error").to_string(
        float_format=lambda x: f"{x:.3f}"
    )
)

# Persist the full per-row error breakdown for later inspection.
error_df.to_csv(RESULTS_DIR / "validation_errors.csv", index=True)


# ------------------------------------------------------------
# Plot 1: Predicted versus actual values
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

# Scatter of actual (x) vs predicted (y) values; semi-transparent points
# (alpha=0.5) make overlapping/dense regions visible.
ax.scatter(y_actual, y_pred, alpha=0.5)

# Determine shared axis limits so the 1:1 reference line spans the
# full range of both actual and predicted values.
lower = min(np.min(y_actual), np.min(y_pred))
upper = max(np.max(y_actual), np.max(y_pred))

# Diagonal y = x line: points falling exactly on this line represent
# perfect predictions (predicted == actual).
ax.plot(
    [lower, upper],
    [lower, upper],
    "k--",
    label="Perfect prediction",
)

ax.set_xlabel("Actual HCHO (µg/m³)")
ax.set_ylabel("Predicted HCHO (µg/m³)")
ax.set_title("Validation: Predicted versus Actual HCHO")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "predicted_vs_actual.png", dpi=300)
plt.close(fig)  # Free the figure's memory now that it's saved.


# ------------------------------------------------------------
# Plot 2: Residuals versus predicted values
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 5))

# Scatter of predicted value (x) vs residual (y). Ideally residuals
# should scatter randomly around zero with no obvious trend or
# funnel shape (which would indicate bias or heteroscedasticity).
ax.scatter(y_pred, error_df["Residual"], alpha=0.5)

# Zero-residual reference line: points on this line have no error.
ax.axhline(0, color="black", linestyle="--")

ax.set_xlabel("Predicted HCHO (µg/m³)")
ax.set_ylabel("Residual: Actual - Predicted (µg/m³)")
ax.set_title("Validation Residuals versus Predictions")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "residuals_vs_predicted.png", dpi=300)
plt.close(fig)

print(f"\nValidation results saved to: {RESULTS_DIR}")


# ------------------------------------------------------------
# Reconstruct missing formaldehyde values
# ------------------------------------------------------------

# Restrict reconstruction to the same sensor used for validation.
sensor_mask = (
    cleaned_df["sensorid"].astype(str) == str(SENSOR_ID)
)

# Rows where the target (Formaldehyde) is missing and therefore a
# candidate for reconstruction/imputation.
missing_hcho = cleaned_df[TARGET].isna()

# Rows where every required feature column has a value (no NaNs),
# i.e. rows the model could actually make a prediction for.
features_available = cleaned_df[FEATURES].notna().all(axis=1)

# Final set of rows to predict: belongs to this sensor, target is
# missing, and all features needed for prediction are present.
rows_to_predict = (
    sensor_mask
    & missing_hcho
    & features_available
)

# Fit the final model on all complete, labelled observations
# for this sensor, not just the training subset.
# (Rows for this sensor where the target IS known and all features
# are available — used to train the model that will fill in the gaps.)
labelled_rows = (
    sensor_mask
    & cleaned_df[TARGET].notna()
    & features_available
)

# Only attempt reconstruction if there is actually something to predict.
if rows_to_predict.any():
    X_all = cleaned_df.loc[labelled_rows, FEATURES]
    y_all = cleaned_df.loc[labelled_rows, TARGET]

    # Guard against attempting to fit a model on too little data.
    if len(X_all) < 2:
        raise ValueError(
            "Not enough labelled observations to reconstruct missing values."
        )

    # Train a fresh model instance on ALL labelled data for this sensor
    # (train + validation combined) to get the best possible model for
    # the final imputation step, rather than reusing the train-only model.
    final_model = make_model()
    final_model.fit(X_all, y_all)

    # Predict Formaldehyde for the rows identified earlier as missing
    # the target but having complete feature data.
    X_missing = cleaned_df.loc[rows_to_predict, FEATURES]
    predictions = final_model.predict(X_missing)

    # Build an output table carrying identifying columns plus the
    # newly predicted (reconstructed) Formaldehyde values.
    reconstructed = cleaned_df.loc[
        rows_to_predict,
        ["id", "sensorid", "createdate"],
    ].copy()

    reconstructed["Predicted HCHO (µg/m³)"] = predictions

    output_path = RESULTS_DIR / "hcho_reconstruction.csv"
    reconstructed.to_csv(output_path, index=False)

    print(
        f"\nReconstructed {len(reconstructed)} missing "
        "formaldehyde values."
    )
    print(f"Saved predictions to: {output_path}")

else:
    # No rows needed reconstruction (either none were missing, or
    # none of the missing rows had complete enough feature data).
    print(
        "\nNo missing formaldehyde rows with all required features "
        f"were found for sensor {SENSOR_ID}."
    )
