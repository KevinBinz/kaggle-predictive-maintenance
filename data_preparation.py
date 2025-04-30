import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from scipy import signal # Import scipy.signal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import pywt

def combine_event_data(errors_df, failures_df, maintenance_df, machines_df):
    """
    Process and combine error, failure, maintenance, and machine dataframes.
    Adds time features (dayofweek, timeofday) and merges machine info (model, age).
    

    Parameters:
    -----------
    errors_df : pandas.DataFrame
        DataFrame containing error events
    failures_df : pandas.DataFrame
        DataFrame containing failure events
    maintenance_df : pandas.DataFrame
        DataFrame containing maintenance events
    machines_df : pandas.DataFrame
        DataFrame containing machine metadata (machineID, model, age)
        
    Returns:
    --------
    pandas.DataFrame
        Combined and enriched DataFrame.
    """
    # Make copies to avoid modifying original dataframes
    errors = errors_df.copy()
    failures = failures_df.copy()
    maintenance = maintenance_df.copy()
    machines = machines_df.copy()
    
    # Rename columns to standardize
    errors.rename(columns={"errorID": "eventcategory"}, inplace=True)
    failures.rename(columns={"failure": "eventcategory"}, inplace=True)
    maintenance.rename(columns={"comp": "eventcategory"}, inplace=True)
    
    # Add eventType column to each dataframe
    errors['eventType'] = 'error'
    failures['eventType'] = 'failure'
    maintenance['eventType'] = 'maintenance'
    
    # Combine all event dataframes
    combined_df = pd.concat([errors, failures, maintenance], ignore_index=True)

    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    combined_df['dayofweek'] = combined_df['datetime'].dt.dayofweek # Monday=0, Sunday=6
    combined_df['timeofday'] = combined_df['datetime'].dt.hour

    combined_df = pd.merge(combined_df, machines, on='machineID', how='left')    

    start_date = f"2015-01-01"
    end_date = f"2016-01-01"
    combined_df = combined_df[(combined_df['datetime'] >= start_date) & (combined_df['datetime'] <= end_date)]

    return combined_df

def merge_concurrent_events(df):
    """
    Merges event records that occur at the exact same datetime for the same
    machine and event category.
    Concatenates eventType for merged records.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame, expected to have columns like 'machineID', 'datetime',
        'eventcategory', 'eventType', and potentially others like 'model', 'age'.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with concurrent events merged.
    """
    print(f"Shape before merging concurrent events: {df.shape}")

    # Define columns to group by
    grouping_cols = ['machineID', 'datetime', 'eventcategory']
    if not all(col in df.columns for col in grouping_cols):
        print("Error: DataFrame missing one or more grouping columns: {grouping_cols}. Skipping merge.")
        return df

    # Define aggregation logic
    agg_dict = {}
    # Aggregate eventType by joining unique sorted values
    if 'eventType' in df.columns:
        agg_dict['eventType'] = lambda x: ', '.join(sorted(x.astype(str).unique()))

    # For other columns, take the first value (assuming they are consistent within the group)
    other_cols = [col for col in df.columns if col not in grouping_cols + ['eventType']]
    for col in other_cols:
        agg_dict[col] = 'first'

    # Perform the aggregation
    try:
        # as_index=False keeps grouping keys as columns
        merged_df = df.groupby(grouping_cols, as_index=False).agg(agg_dict)
    except Exception as e:
        print(f"Error during aggregation: {e}. Returning original DataFrame.")
        return df

    print(f"Shape after merging concurrent events: {merged_df.shape}")
    return merged_df

def _pivot_events_by_category(df):
    """
    Pivots the event DataFrame so each unique eventcategory becomes a column.
    The values in these new columns are the corresponding eventType.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input event DataFrame with columns 'machineID', 'datetime', 
        'eventcategory', 'eventType'. It's recommended to run 
        merge_concurrent_events on the input first.

    Returns:
    --------
    pandas.DataFrame
        A pivoted DataFrame with machineID, datetime as identifiers, 
        eventcategory values as columns, and eventType as values.
    """
    required_cols = ['machineID', 'datetime', 'eventcategory', 'eventType']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input DataFrame missing one or more required columns for pivoting: {required_cols}")
        return pd.DataFrame() # Return empty DataFrame

    print(f"\nPivoting event data by category. Input shape: {df.shape}")

    try:
        # Pivot the table
        pivoted_df = pd.pivot_table(
            df,
            index=['machineID', 'datetime'],
            columns='eventcategory',
            values='eventType',
            aggfunc='first' # Assumes unique eventType per group after potential merging
        )

        # Fill NaN values resulting from pivot with empty strings
        pivoted_df.fillna('', inplace=True)

        # Reset index to make machineID and datetime regular columns
        pivoted_df.reset_index(inplace=True)

        print(f"Pivoted DataFrame shape: {pivoted_df.shape}")
        # Rename the columns index (originally 'eventcategory') to None for cleaner look
        pivoted_df.columns.name = None

    except Exception as e:
        print(f"Error during pivoting: {e}")
        return pd.DataFrame()

    return pivoted_df

def curate_pivoted_events(df):
    """
    Adds curated features to a pivoted event DataFrame.

    Parameters:
    -----------
    pivoted_df : pandas.DataFrame
        Input pivoted DataFrame from pivot_events_by_category. 
        Must include 'machineID', 'datetime', and columns for event categories.

    Returns:
    --------
    pandas.DataFrame
        The DataFrame with added columns: 'is_6am', 'isMaintenanceEvent', 
        'isErrorEvent', 'isFailureEvent', 'daysSinceLastMaintenanceEvent', 
        'isScheduled', 'CuratedEventType'.
    """
    df = df.copy()

    df = _pivot_events_by_category(df)

    required_cols = ['machineID', 'datetime']
    if not all(col in df.columns for col in required_cols):
        print("Error: Pivoted DataFrame missing 'machineID' or 'datetime'. Cannot curate.")
        return df

    print(f"\nCurating pivoted events. Input shape: {df.shape}")

    # Ensure datetime is in the correct format
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
            print(f"Error converting 'datetime' column to datetime objects: {e}")
            return df # Return original on error

    # 1. Add is_6am
    df['is_6am'] = (df['datetime'].dt.hour == 6)

    # Identify component and error columns dynamically
    comp_cols = [col for col in df.columns if str(col).startswith('comp')]
    error_cols = [col for col in df.columns if str(col).startswith('error')]

    print(f"Identified Component Columns: {comp_cols}")
    print(f"Identified Error Columns: {error_cols}")

    # 2. Add isMaintenanceEvent (any comp column is non-empty)
    if comp_cols:
        df['isMaintenanceEvent'] = (df[comp_cols] != '').any(axis=1)
    else:
        print("Warning: No component columns found. 'isMaintenanceEvent' set to False.")
        df['isMaintenanceEvent'] = False

    # 3. Add isErrorEvent (any error column is non-empty)
    if error_cols:
        df['isErrorEvent'] = (df[error_cols] != '').any(axis=1)
    else:
        print("Warning: No error columns found. 'isErrorEvent' set to False.")
        df['isErrorEvent'] = False

    # 4. Add isFailureEvent (any comp column contains 'failure')
    if comp_cols:
        # Apply str.contains safely
        failure_checks = df[comp_cols].apply(lambda col: col.astype(str).str.contains('failure', na=False))
        df['isFailureEvent'] = failure_checks.any(axis=1)
    else:
        print("Warning: No component columns found. 'isFailureEvent' set to False.")
        df['isFailureEvent'] = False

    # 5. Calculate daysSinceLastMaintenanceEvent
    # Ensure sorting
    df.sort_values(by=['machineID', 'datetime'], inplace=True)

    # Get timestamps of maintenance events only
    maint_times = df['datetime'].where(df['isMaintenanceEvent'])

    # Shift these timestamps within each machine group to get the *previous* maint time
    # Then forward fill this previous time to subsequent rows
    df['last_maint_time'] = maint_times.groupby(df['machineID']).shift(1).ffill()

    time_diff_maint = df['datetime'] - df['last_maint_time']
    # Calculate as float days, NaNs will appear before first maintenance
    df['daysSinceLastMaintenanceEvent'] = time_diff_maint.dt.total_seconds() / (24 * 3600)
    df.drop(columns=['last_maint_time'], inplace=True) # Drop temporary column

    # 6. Add isScheduled column
    days_since_maint = df['daysSinceLastMaintenanceEvent']
    # isScheduled is True if days_since_maint > 0 and divisible by 15. NaNs become False.
    df['isScheduled'] = ((days_since_maint > 0) & (days_since_maint % 15 == 0)).fillna(False)

    # 7. Create CuratedEventType
    def determine_curated_type(row):
        parts = []
        if row['isMaintenanceEvent']: parts.append('Maintenance')
        if row['isErrorEvent']:       parts.append('Error')
        if row['isFailureEvent']:     parts.append('Failure')
        # Return hyphen-separated string or empty if none are true
        return '- '.join(parts)

    df['CuratedEventType'] = df.apply(determine_curated_type, axis=1)

    df = _track_error_history(df)

    print(f"Curated pivoted DataFrame shape: {df.shape}")
    return df

def join_events_and_telemetry(curated_pivoted, telemetry_df):
    """Performs a left join between telemetry data and pivoted event data.

    Args:
        curated_pivoted (pd.DataFrame): DataFrame with pivoted event data, indexed by datetime.
        telemetry_df (pd.DataFrame): DataFrame with telemetry data, containing a 'datetime' column.

    Returns:
        pd.DataFrame: The result of the left join, with telemetry data on the left.
                    Event columns will be None/NaN for telemetry samples with no matching event.
    """ 

    telemetry_df.to_csv('telemetry_df.csv', index=False)
    # curated_pivoted.to_csv('curated_pivoted_reset.csv', index=False)

    # Print the first few rows of the telemetry and curated pivoted dataframes
    print(f"Telemetry shape before join: {telemetry_df.shape}")
    print(f"Curated pivoted shape before join: {curated_pivoted.shape}")

    # Perform the full outer merge with indicator
    merged_df = pd.merge(
        telemetry_df,
        curated_pivoted,
        on=['datetime', 'machineID'],
        how='left',
        indicator=True # Adds a '_merge' column
    )

    # --- Calculate and Print Join Statistics ---
    total_records = len(merged_df)
    telemetry_only = len(merged_df[merged_df['_merge'] == 'left_only'])
    event_only = len(merged_df[merged_df['_merge'] == 'right_only'])
    both = len(merged_df[merged_df['_merge'] == 'both'])

    print("\n--- Join Statistics (Telemetry vs Events on 'datetime') ---")
    print(f"Total records in merged data: {total_records}")
    print(f"Records with only telemetry data: {telemetry_only}")
    print(f"Records with only event data: {event_only}")
    print(f"Records with both telemetry and event data: {both}")
    print("---------------------------------------------------------")

    # Drop the indicator column if not needed downstream
    merged_df.drop(columns=['_merge'], inplace=True)

    print(f"Joined DataFrame shape: {merged_df.shape} columns: {merged_df.columns.tolist()}")

    return merged_df

def impute_error_counts(joined_df):
    """
    Imputes NaN values in 'CountOfErrorXSinceLastMaintenance' columns using backward fill
    within each machine group.

    Parameters:
    -----------
    joined_df : pandas.DataFrame
        DataFrame potentially containing NaN values in error count columns.
        Must have 'machineID' and 'datetime' columns.

    Returns:
    ---------
    pandas.DataFrame
        DataFrame with imputed error count columns.
    """
    if not all(col in joined_df.columns for col in ['machineID', 'datetime']):
        print("Error: Input DataFrame missing 'machineID' or 'datetime'. Cannot impute.")
        return joined_df

    df = joined_df.copy()
    print(f"\nImputing error count columns. Input shape: {df.shape}")

    # Identify columns to impute
    cols_to_impute = [col for col in df.columns if col.startswith('CountOf') and col.endswith('SinceLastMaintenance')]

    if not cols_to_impute:
        print("No 'CountOfError...SinceLastMaintenance' columns found to impute.")
        return df

    print(f"Columns to impute: {cols_to_impute}")

    # IMPORTANT: Ensure data is sorted for bfill within groups
    df.sort_values(by=['machineID', 'datetime'], inplace=True)

    # Create temporary columns for imputation
    imputed_cols_data = {}
    for col in cols_to_impute:
        # Copy original data
        temp_col_data = df[col].copy()
        # Set counts on maintenance rows to NaN so they don't forward fill
        temp_col_data.loc[df['isMaintenanceEvent'] == True] = np.nan
        imputed_cols_data[col] = temp_col_data

    temp_impute_df = pd.DataFrame(imputed_cols_data, index=df.index)

    # Perform forward fill within each machine group on the temp data
    print("Performing forward fill (ffill) within each machine group, resetting after maintenance...")
    imputed_ffilled = temp_impute_df.groupby(df['machineID']).ffill()

    # Fill remaining NaNs (start of history or after maintenance before new error) with 0
    print("Filling NaNs (at start/after maintenance) with 0...")
    imputed_filled_zero = imputed_ffilled.fillna(0)

    # Update the original DataFrame columns
    df[cols_to_impute] = imputed_filled_zero

    # Optional: Convert imputed columns to integer type if desired (and appropriate)
    # try:
    #     df[cols_to_impute] = df[cols_to_impute].astype(int)
    # except ValueError as e:
    #     print(f"Warning: Could not convert imputed columns to int: {e}")

    print(f"Imputation complete. Shape: {df.shape}")
    return df

def localize_relevant_events(joined_df):
    """Adds columns for time-to-next-failure and time-since-last-component-maintenance.

    Args:
        joined_df (pd.DataFrame): DataFrame containing telemetry and event data,
                                   sorted by machineID and datetime.

    Returns:
        pd.DataFrame: DataFrame with added 'NextFailureTimestamp' and 
                      'TimeSinceLastMaintenanceCompN' columns.
    """
    if not all(col in joined_df.columns for col in ['machineID', 'datetime', 'isFailureEvent']):
        print("Error: DataFrame missing 'machineID', 'datetime', or 'isFailureEvent'. Cannot localize events.")
        return joined_df

    df = joined_df.copy()
    print(f"\nLocalizing relevant events. Input shape: {df.shape}")

    # --- Calculate NextFailureTimestamp --- 
    # Ensure sorting (should already be done, but defensive)
    df.sort_values(by=['machineID', 'datetime'], inplace=True)

    # Get timestamps of failure events only
    failure_times = df['datetime'].where(df['isFailureEvent'] == True)

    # Backward fill failure times within each machine group
    df['NextFailureTimestamp'] = failure_times.groupby(df['machineID']).bfill()
    print("Added 'NextFailureTimestamp'.")

    # --- Calculate TimeSinceLastMaintenanceCompN --- 
    comp_cols = [f'comp{i}' for i in range(1, 5)]
    temp_cols_to_drop = []

    for comp_name in comp_cols:
        if comp_name not in df.columns:
            print(f"Warning: Column '{comp_name}' not found. Skipping TimeSinceLastMaintenance for it.")
            continue
            
        # Identify timestamps where this component had maintenance/failure (non-empty)
        comp_event_times = df['datetime'].where(df[comp_name] != '')
        
        # Forward fill these timestamps within each machine group
        last_comp_event_col = f'_LastMaintTimestamp_{comp_name}'
        df[last_comp_event_col] = comp_event_times.groupby(df['machineID']).ffill()
        temp_cols_to_drop.append(last_comp_event_col)
        
        # Calculate time difference in days
        time_diff = df['datetime'] - df[last_comp_event_col]
        new_col_name = f'TimeSinceLastMaintenance{comp_name.capitalize()}' # e.g., TimeSinceLastMaintenanceComp1
        df[new_col_name] = time_diff.dt.total_seconds() / (24 * 3600)
        print(f"Added '{new_col_name}'.")

    # Clean up temporary timestamp columns
    df.drop(columns=temp_cols_to_drop, inplace=True, errors='ignore')

    print(f"Finished localizing events. Shape: {df.shape}")
    return df

def _track_error_history(curated_pivoted_df):
    """
    Adds a column counting specific error occurrences since the last maintenance event.
    The count resets after a maintenance event.

    Parameters:
    -----------
    curated_pivoted_df : pandas.DataFrame
        DataFrame output from curate_pivoted_events, containing 'machineID', 
        'datetime', 'isMaintenanceEvent', and pivoted error columns like 'error1',
        'error2', 'error3', 'error4', 'error5'.

    Returns:
    ---------
    pandas.DataFrame
        The DataFrame with added count columns for errors 1 through 5 (if present)
        since the last maintenance.
    """
    # Check for essential columns first
    base_required_cols = ['machineID', 'datetime', 'isMaintenanceEvent']
    if not all(col in curated_pivoted_df.columns for col in base_required_cols):
        print(f"Error: Input DataFrame missing base required columns for error tracking: {base_required_cols}")
        return curated_pivoted_df

    # Check which error columns are present
    error_cols_present = []
    for i in [1, 2, 3, 4, 5]:
        col_name = f'error{i}'
        if col_name in curated_pivoted_df.columns:
            error_cols_present.append(col_name)
        else:
            print(f"Warning: Column '{col_name}' not found, skipping its count.")

    if not error_cols_present:
        print("Error: No error columns (error1 through error5) found to track.")
        return curated_pivoted_df

    df = curated_pivoted_df.copy()
    print(f"\nTracking Error history. Input shape: {df.shape}")

    # 1. Ensure Sort Order
    df.sort_values(by=['machineID', 'datetime'], inplace=True)

    # 2. Identify error occurrences for present columns
    temp_error_cols = []
    for col_name in error_cols_present:
        temp_col = f'_is{col_name.capitalize()}'
        # Check if column is already boolean, otherwise check non-emptiness
        if pd.api.types.is_bool_dtype(df[col_name]):
             df[temp_col] = df[col_name]
        else:
             df[temp_col] = (df[col_name] != '')
        temp_error_cols.append(temp_col)

    # --- Add check and temporary column for double error (error2 and error3) --- 
    temp_double_error_col = '_isDoubleError2_3'
    double_error_possible = False
    if 'error2' in df.columns and 'error3' in df.columns:
        # Check if boolean or non-empty string
        error2_check = df['error2'] if pd.api.types.is_bool_dtype(df['error2']) else (df['error2'] != '')
        error3_check = df['error3'] if pd.api.types.is_bool_dtype(df['error3']) else (df['error3'] != '')
        df[temp_double_error_col] = error2_check & error3_check
        double_error_possible = True
        print("Tracking double error (error2 & error3) occurrences.")
    else:
        print("Warning: Columns 'error2' and/or 'error3' not present. Cannot track double errors.")
    # --------------------------------------------------------------------------

    # 3. Identify Maintenance Blocks
    # Shift ensures the maintenance row is the *last* row of its block
    df['_maint_block'] = df.groupby('machineID')['isMaintenanceEvent'].shift(1, fill_value=False).cumsum()

    # 4. Calculate Cumulative Counts within Blocks
    groupby_cols = ['machineID', '_maint_block']
    
    # Calculate for individual errors present
    for i, col_name in enumerate(error_cols_present):
        temp_col = temp_error_cols[i]
        new_col_name = f'CountOf{col_name.capitalize()}SinceLastMaintenance'
        df[new_col_name] = df.groupby(groupby_cols)[temp_col].cumsum()
        print(f"Added '{new_col_name}'.")

    # Calculate for double error if possible
    if double_error_possible:
        new_double_col_name = 'CountOfDoubleErrorsSinceLastMaintenance'
        df[new_double_col_name] = df.groupby(groupby_cols)[temp_double_error_col].cumsum()
        print(f"Added '{new_double_col_name}'.")

    # 5. Clean up temporary columns
    cols_to_drop = temp_error_cols + ['_maint_block']
    if double_error_possible:
        cols_to_drop.append(temp_double_error_col)
        
    df.drop(columns=cols_to_drop, inplace=True)

    print(f"Finished tracking errors. Shape: {df.shape}")
    return df

def add_basic_lag_stats(joined_df):
    """
    Calculates rolling window statistics for telemetry data.

    Args:
        joined_df (pd.DataFrame): DataFrame with telemetry data, 
        including 'datetime', 'machineID', 
        'volt', 'rotate', 'pressure', 'vibration'.

    Returns:
        pd.DataFrame: DataFrame with added rolling window statistics columns.
    """
    df = joined_df.copy()

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
            print(f"Error converting 'datetime': {e}. Cannot proceed.")
            return joined_df # Return original

    # Sort by machine and time for processing order
    df = df.sort_values(by=['machineID', 'datetime'])

    all_machines_features = [] # List to store feature DataFrames for each machine

    unique_machines = df['machineID'].unique()

    print(f"Processing {len(unique_machines)} machines...")

    for machine_id in unique_machines:
        df_machine = df[df['machineID'] == machine_id].copy()
        df_machine = df_machine.set_index('datetime').sort_index()

        machine_features_list = [] # Features for this specific machine

        if df_machine.empty:
            continue # Should not happen if unique_machines comes from df, but safe check

        for win in ['1h', '6h', '12h', '24h', '72h', '168h']:
            for col in ['volt', 'rotate', 'pressure', 'vibration']:
                # Perform rolling calculation on the single-machine DataFrame (DatetimeIndex)
                rolled_aggs = df_machine[col].rolling(window=win, min_periods=1).agg(['min', 'median', 'mean', 'max', 'var'])

                # Rename the aggregate columns
                rolled_aggs = rolled_aggs.rename(columns=lambda x: f'{col}_{win}_{x}')

                # Append the calculated features DataFrame to the machine's list
                machine_features_list.append(rolled_aggs)

        # Concatenate all features for the current machine
        if machine_features_list:
            machine_features_df = pd.concat(machine_features_list, axis=1)
            # Add machineID back (useful if index is reset later, though join uses index)
            machine_features_df['machineID'] = machine_id
            # Append to the main list
            all_machines_features.append(machine_features_df)

    print("Finished calculating features for all machines.")

    # Concatenate all feature DataFrames at once
    print("Concatenating all calculated features...")
    if all_machines_features:
        features_df = pd.concat(all_machines_features)
        # Prepare features_df for merge (needs machineID and datetime index)
        features_df = features_df.reset_index().set_index(['machineID', 'datetime'])

        # Prepare original df for merge (needs machineID and datetime index)
        df_original_indexed = joined_df.set_index(['machineID', 'datetime'])

        # Merge the features back to the original DataFrame
        print(f"Joining features (shape={features_df.shape}) back to the main DataFrame...")
        df_merged = df_original_indexed.join(features_df, how='left')
        df = df_merged.reset_index() # Reset index to get columns back
    else:
        print("No rolling features were calculated.")
        df = joined_df # Return original if nothing was calculated

    print(f"Shape before: {joined_df.shape} after {df.shape}")
    return df

def save_df(fn, df):
    output_csv_dir = "data"
    os.makedirs(output_csv_dir, exist_ok=True)
    df.to_csv(os.path.join(output_csv_dir, fn), index=False)
    print(f"\nSaved event data to {fn}")

def process_data():
    data_path = "telemetry"
       
    # Define file paths
    errors_file = os.path.join(data_path, "PdM_errors.csv")
    failures_file = os.path.join(data_path, "PdM_failures.csv")
    maintenance_file = os.path.join(data_path, "PdM_maint.csv")
    machines_file = os.path.join(data_path, "PdM_machines.csv") 
    telemetry_file = os.path.join(data_path, "PdM_telemetry.csv")

    # --- Load Data --- 
    PDM_Errors = pd.read_csv(errors_file, parse_dates=['datetime'])
    Failures = pd.read_csv(failures_file, parse_dates=['datetime'])
    Maintenance = pd.read_csv(maintenance_file, parse_dates=['datetime'])
    PDM_Machines = pd.read_csv(machines_file) 
    PDM_Telemetry = pd.read_csv(telemetry_file, parse_dates=['datetime']) # Load telemetry
    
    print("Data loaded successfully.")
   
    # --- Process Event Data --- 
    events = combine_event_data(PDM_Errors, Failures, Maintenance, PDM_Machines)
    curated_events = curate_pivoted_events(events)
    save_df(fn="curated_events.csv", df=curated_events)

    curated_telemetry = add_basic_lag_stats(PDM_Telemetry)
    save_df(fn="curated_telemetry.csv", df=curated_telemetry)

    joined_df = join_events_and_telemetry(curated_events, curated_telemetry)
    imputed_df = impute_error_counts(joined_df)
    final_df = localize_relevant_events(imputed_df)
    save_df(fn="final.csv", df=final_df)

    return curated_events, joined_df, final_df # Return the final df

# Example usage
if __name__ == "__main__":
    process_data()

   
