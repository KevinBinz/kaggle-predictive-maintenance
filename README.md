# Predictive Maintenance Machine Learning Pipeline

This repository contains a complete machine learning pipeline for predictive maintenance using sensor data from industrial machines. The project predicts machine failures based on telemetry data, error logs, and maintenance records.

## Project Structure

- **Data Processing Scripts:**
  - `eda_event.py`: Processes event data (errors, failures, maintenance)
  - `eda_telemetry.py`: Processes telemetry sensor data 
  - `modeling.py`: Builds and evaluates machine learning models

- **Data Flow:**
  1. Raw data files in `telemetry/` directory
  2. Processed intermediate files in `data/` directory
  3. Visualizations stored in `plots/` directory

## Data Sources

The project uses several datasets in the `telemetry/` directory:

- `PdM_telemetry.csv`: Time series sensor data (volt, rotate, pressure, vibration)
- `PdM_errors.csv`: Error logs from machines
- `PdM_failures.csv`: Records of machine failures
- `PdM_maint.csv`: Maintenance records
- `PdM_machines.csv`: Machine metadata (ID, model type, age)

## Feature Engineering

The data processing pipeline includes extensive feature engineering:

### Time-based Features
- **Basic time features:** Added dayofweek, timeofday (hour) from datetime
- **Morning indicator:** Added `is_6am` flag to capture maintenance patterns

### Event-Based Features 
- **Component indicators:** Binary flags for components (comp1-comp4)
- **Error indicators:** Binary flags for error types (error1-error5)
- **Event type classification:** Created `CuratedEventType` to categorize events

### Rolling Window Aggregations
- Created lagged features for all sensor measurements with multiple window sizes:
  - 1-hour windows: Capturing immediate machine state
  - 168-hour (weekly) windows: Capturing longer-term patterns
- For each window and sensor, calculated multiple statistics:
  - min, median, mean, max, variance

### Machine Metadata Integration
- Added machine `model` type (categorical)
- Added machine `age` (numerical)

## Model Training Process

The modeling pipeline:

1. **Data Splitting:** Time-based split (train: Jan-Oct 2015, test: Nov-Dec 2015)
2. **Feature Selection:** Removes non-predictive features (too informative/leaky)
3. **H2O AutoML:** Automatically trains and tunes multiple model types
4. **Model Evaluation:** Compares performance metrics (AUC, LogLoss, Accuracy)
5. **Feature Importance Analysis:** Saves importance scores to `data/feature_importances.csv`

## Running the Pipeline

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the data processing scripts:
   ```
   python eda_event.py
   python eda_telemetry.py 
   ```

3. Run the modeling script:
   ```
   python modeling.py
   ```

## Model Results

The project uses H2O AutoML to find the best model. The final model:
- Generates cross-validation metrics using 5-fold CV
- Handles missing values using mean imputation
- Outputs performance metrics for both training and test data
- Provides detailed confusion matrices for error analysis 

## Prediction Visualization Example (Machine 1, Event 2015-01-05)

![Machine 1 Telemetry Around Failure Event 2015-01-05](plots/prediction_viz_machine_1_event_2015-01-05.png) 