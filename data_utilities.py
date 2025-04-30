import pandas as pd

def prepare_binary_classifier_trainset(only_maintenance, agg_list, lag_list):
    df = pd.read_csv('data/final.csv')
    # Adding model_type and machine_age to the features
    df = df.merge(pd.read_csv('telemetry/PdM_machines.csv'), on='machineID', how='left')

    # Filter
    print(f"df.shape={df.shape}")
    if only_maintenance:
        df = df[df['isMaintenanceEvent']==True]
    else: 
        df = df[df['isMaintenanceEvent']!=True]
    print(f"Filter only_maintenance={only_maintenance}. New df.shape={df.shape}")

    # Remove columns unsuitable for training
    trivial = ['isMaintenanceEvent', 'isErrorEvent']
    too_informative = ['comp1', 'comp2', 'comp3', 'comp4', 'error1', 'error2', 'error3', 'error4', 'error5', 'is_6am', 'CuratedEventType', 'NextFailureTimestamp']
    exclude_columns = get_exclude_columns(df.columns.to_list(),  lag_list=lag_list, agg_list=agg_list)
    drop_cols = exclude_columns + too_informative + trivial
    df.drop(columns=drop_cols, inplace=True)
    print(f"Dropped cols: {drop_cols}")
    print(f"Remaining cols: {df.columns.tolist()}")
    print(f"After dropping {len(drop_cols)} columns, new df.shape={df.shape}")

    # Display df metadata, and save to csv
    null_counts = df.isnull().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    print(f"Nulls: {columns_with_nulls}")

    print(f"Value Counts: {df['isFailureEvent'].value_counts()}")
    fn = "only_maintenance" if only_maintenance else "non_maintenance"
    df.to_csv(f"data/{fn}.csv", index=False)
    return df

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

def get_exclude_columns(list_columns, lag_list, agg_list):
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
            elif parts[2] in agg_list:
                selected_columns.append(col)
                
    print(f"Selected {len(selected_columns)} columns for lags {lag_list}h.")
    return selected_columns