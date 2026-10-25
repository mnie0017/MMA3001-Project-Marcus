

# Formaldehyde Concentration Reconstruction Using Environmental Sensor Data

A Python-based project investigating the reconstruction of missing formaldehyde concentration measurements using environmental sensor data through the use of data processing and neural network modelling.

## Problem Statement and Relevance

Formaldehyde is an air pollutant that can pose health risks when inhaled at high levels. Reliable monitoring is important for assessing indoor air quality and identifying conditions where additional ventilation may be required to prevent unsafe concentrations.

Environmental sensors can produce incomplete data due to missing readings or measurement interruptions. Reconstructing missing formaldehyde measurements using other available sensor variables could help maintain continuity in air quality monitoring and identify potentially unsafe conditions.

The aim of this project is to develop computational models that estimate formaldehyde concentrations from other environmental measurements to ensure continous monitoring and thus saftey.
## Features

- **Data auditing and preparation:** Examines and prepares environmental sensor data for modelling.
- **Data splitting:** Separates observations into training and validation sets.
- **Neural network modelling:** Uses a multilayer perceptron to learn relationships between environmental variables and formaldehyde concentration.
- **Model evaluation:** Evaluates predictions using mean absolute error (MAE), root mean squared error (RMSE), and the coefficient of determination (R^2).
- **Data reconstruction:** Produces estimated formaldehyde concentrations using the trained model.

## Input Data

The project uses environmental sensor measurements from the supplied dataset.

The target variable is:

- `Formaldehyde (µg/m3)`

The neural network uses the following input features:

- `Temperature (°C)`
- `Humidity (HR%)`
- `Carbon dioxide (ppm)`
- `Light Volatile Organic Compounds (ppb)`
- `Particulate matter 1 (µg/m3)`
- `Particulate matter 2.5 (µg/m3)`

The analysis selects sensor ID `6012002000869`but other IDs can be inputed by swapping this value with another ID in the data set.

The raw dataset is not included in the repository and must be obtained separately before running the data-processing workflow.

## Project Structure

```text
MMA3001-Project-Marcus/
├── README.md
├── .gitignore
├── src/
│   ├── Auditing.py
│   ├── Data_split.py
│   └── NeuralNetwork.py
├── results/
├── tests/
└── docs/
```

- `Auditing.py`: Data auditing and preparation.
- `Data_split.py`: Functions for splitting the dataset into training and validation sets.
- `NeuralNetwork.py`: Neural network training, prediction, and evaluation.
- `results/`: Generated performance metrics and reconstructed data.
- `tests/`: Location for pytests.
- `docs/`: Project documentation.

## Installation

The project is written in Python and uses the following scientific computing and machine learning libraries:

- NumPy
- pandas
- Matplotlib
- scikit-learn

Install the required packages using:

pip install numpy pandas matplotlib scikit-learn

The project can be run in Google Colab or another Python environment with the required dependencies installed.

## How to Run the Project

### 1. Obtain the repository

Clone the repository:


git clone https://github.com/mnie0017/MMA3001-Project-Marcus.git
cd MMA3001-Project-Marcus


### 2. Prepare the dataset
This is what was used to setup colab to use the data 

# Creating folders
!mkdir -p src data tests docs
!touch src/.gitkeep tests/.gitkeep docs/.gitkeep
!echo "data/*" >> .gitignore
# Obtain data from google drive
from google.colab import drive
drive.mount("/content/drive")
!cp "/content/drive/MyDrive/MMA3001 project/5EnvSensor_MayToDec2024_180kRows.zip" /content/MMA3001-Project-Marcus/data/
!ls -lh /content/MMA3001-Project-Marcus/data
# Unzipping files in data
%cd /content/MMA3001-Project-Marcus/data
!unzip -o "5EnvSensor_MayToDec2024_180kRows.zip"
%cd /content/MMA3001-Project-Marcus
!ls -lh data

### 3. Audit and prepare the data

Run `Auditing.py` to perform the required data auditing and preparation.

### 4. Split the data

Use the functions in `Data_split.py` to separate the observations into training and validation sets.

### 5. Train and evaluate the model

Run `NeuralNetwork.py` after completing the data preparation and splitting steps. The model uses the environmental input features to predict formaldehyde concentration.

The workflow includes feature standardisation, neural network training, prediction, and performance evaluation. The enviromental inputs can be limited further if more than just formaldehyde values are missing. However, analysis with a 2 variable neural network showed a significant decrease in performance.
Thus the user should be cautious about reducing the input varaibles. Sensor ID can be changed to reconstruct a different sensor in the data. By default the sensor ID is 6012002000869.

## Output Files

The project outputs:

- `results/predicted_vs_actual.png`: Plot of predicted versus actual formaldehyde concentrations.
- `results/residuals_vs_predicted.png`: Plot of residuals versus predicted formaldehyde concentrations.
- `results/hcho_reconstruction.csv`: Reconstructed formaldehyde concentrations.
- `results/validation_errors.csv`: Observation error.
- `results/validation_metrics.csv`: Validation metrics.

## Limitations

- Model performance depends on the quality and representativeness of the available sensor data. For the chosen data set, there were limited unique formaldehyde values avaliable for each sensor.
- Relationships learned are sensor dependent and can't be generalsied  another sensors. The project must be run with a different sensor selected if a different sensor needs to be reconstructed.
- A random training-validation split may not accurately represent performance when predicting future measurements.
- Prediction errors may be larger at unusually high formaldehyde concentrations.
- Reconstructed concentrations are estimates and should not be treated as direct measurements or used as the sole basis for safety-critical ventilation decisions.
- Code was generated using AI due to limited python knowlage. Code was corss referenced with provided lecture notes but errors in the code may still be present.

## Licence
The source code in this repository is licensed under the MIT
Licence.
The sensor dataset used by this project is not included in this
repository and is subject to its own ownership and usage
conditions. The MIT Licence does not apply to that dataset.

## Acknowledgements

This project was developed as part of the MMA3001 project.
