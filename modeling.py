import pandas as pd
import numpy as np
import os
import h2o
from h2o.automl import H2OAutoML

def split_data_by_date(df):
    """Splits DataFrame into training and testing sets based on date.

    Training data: Jan 2015 - Oct 2015
    Testing data: Nov 2015 - Dec 2015

    Args:
        df (pd.DataFrame): Input DataFrame containing 'datetime' and the label column.

    Returns:
        tuple: (train_df, test_df)
    """
    df_copy = df.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])

    # --- Define Date Ranges ---
    train_start = pd.Timestamp('2015-01-01')
    train_end = pd.Timestamp('2015-10-31 23:59:59')
    test_start = pd.Timestamp('2015-11-01')
    test_end = pd.Timestamp('2015-12-31 23:59:59')

    # --- Split Data ---
    train_mask = (df_copy['datetime'] >= train_start) & (df_copy['datetime'] <= train_end)
    test_mask = (df_copy['datetime'] >= test_start) & (df_copy['datetime'] <= test_end)

    train_df = df_copy[train_mask]
    test_df = df_copy[test_mask]

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    return train_df, test_df

def get_columns(list_columns, lag_list):
    """Filters a list of columns to select specific rolling window features.

    Args:
        list_columns (list): The full list of column names.
        lag_list (list, optional): A list of integer hour windows (e.g., [24, 168]). 

    Returns:
        list: A list of column names corresponding to the specified lag windows.
    """
    selected_columns = []
    # Convert lag_list to the string format used in column names (e.g., 24 -> '24h')
    lag_strings = [f"{lag}h" for lag in lag_list]
    
    base_features = ['volt', 'rotate', 'pressure', 'vibration']
    aggs = ['min', 'median', 'mean', 'max', 'var']

    for col in list_columns:
        parts = col.split('_')
        # Check if column matches the pattern: {feature}_{window}h_{agg}
        if len(parts) == 3 and parts[0] in base_features and parts[2] in aggs:
            window_part = parts[1]
            if window_part in lag_strings:
                selected_columns.append(col)
                
    print(f"Selected {len(selected_columns)} columns for lags {lag_list}h.")
    return selected_columns

if __name__ == "__main__":
    output_csv_dir = "data"

    df = pd.read_csv('data/imputed_joined_df.csv')
    only_maintenance = df[df['isMaintenanceEvent']==True]

    trivial = ['isMaintenanceEvent', 'isErrorEvent']
    only_maintenance.drop(columns=trivial, inplace=True)

    too_informative = ['comp1', 'comp2', 'comp3', 'comp4', 'error1', 'error2', 'error3', 'error4', 'error5', 'is_6am', 'CuratedEventType']
    only_maintenance.drop(columns=too_informative, inplace=True)

    # Adding model_type and machine_age to the features
    only_maintenance = only_maintenance.merge(pd.read_csv('telemetry/PdM_machines.csv'), on='machineID', how='left')

    null_counts = only_maintenance.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    print(f"Nulls: {columns_with_nulls}")
    print(f"Shape: {only_maintenance.shape}")

    # features = ['machineID','datetime','volt','rotate','pressure','vibration','isMaintenanceEvent', 'isErrorEvent', 'isFailureEvent']
    # only_maintenance = only_maintenance[features]

    print(only_maintenance['isFailureEvent'].value_counts())

    only_maintenance.to_csv('data/only_maintenance.csv', index=False)

    train_df, test_df = split_data_by_date(only_maintenance)

    h2o.init()
    target = "isFailureEvent"
    exclude_columns = get_columns(only_maintenance.columns.to_list(), lag_list=[1, 168])

    train_hf = h2o.H2OFrame(train_df)
    test_hf = h2o.H2OFrame(test_df)
    train_hf[target] = train_hf[target].asfactor()
    test_hf[target] = test_hf[target].asfactor()
    
    x = [col for col in train_hf.columns if col not in [target] + exclude_columns]
    print(f"excluding {len(exclude_columns)}. final width={len(x)}")

    y = target

    aml = H2OAutoML(
        max_models=10,
        seed=42,
        max_runtime_secs=7200,
        sort_metric='accuracy'
    )
    aml.train(x=x, y=y, training_frame=train_hf)
    print(aml.leaderboard)

    leader = aml.leader
    ftimp_df = leader.varimp(use_pandas=True)
    ftimp_df.to_csv('data/feature_importances.csv', index=False)

    model_summary = leader.summary()
    print(model_summary)

    # Get model details to see preprocessing information
    print("\n--- Feature Preprocessing Information ---")
    model_details = h2o.get_model(leader.model_id)
    # Print specific model parameters
    print("\n--- Selected Model Parameters ---")
    if hasattr(leader, 'params'):
        print(f"Ignored Columns: {leader.params['ignored_columns']['actual']}")
        print(f"Number of Folds: {leader.params['nfolds']['actual']}")
        print(f"Missing Values Handling: {leader.params['missing_values_handling']['actual']}")
        print(f"Balance Classes: {leader.params['balance_classes']['actual']}")
    
    pred = leader.predict(test_hf)

    # --- Evaluate Performance --- 
    print("\n--- Leader Model Performance Comparison ---")
    train_perf = leader.model_performance(train=True)
    test_perf = leader.model_performance(test_data=test_hf)

    print(f"Metric           | Train     | Test")
    print(f"-----------------|-----------|-----------")
    print(f"AUC              | {train_perf.auc():<9.4f} | {test_perf.auc():<9.4f}")
    print(f"LogLoss          | {train_perf.logloss():<9.4f} | {test_perf.logloss():<9.4f}")
    # Accuracy needs careful indexing as it might be nested
    train_acc = train_perf.accuracy()[0][1] if train_perf.accuracy() and len(train_perf.accuracy()) > 0 else np.nan
    test_acc = test_perf.accuracy()[0][1] if test_perf.accuracy() and len(test_perf.accuracy()) > 0 else np.nan
    print(f"Accuracy         | {train_acc:<9.4f} | {test_acc:<9.4f}")

    print("\n--- Confusion Matrix (Train) ---")
    print(train_perf.confusion_matrix())

    print("\n--- Confusion Matrix (Test) ---")
    print(test_perf.confusion_matrix())
