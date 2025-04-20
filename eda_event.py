import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from scipy import signal # Import scipy.signal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import pywt

def process_event_data(errors_df, failures_df, maintenance_df, machines_df):
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

    # Ensure datetime column is in datetime format before extracting features
    if 'datetime' in combined_df.columns:
        combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
        
        # Add time-based features
        combined_df['dayofweek'] = combined_df['datetime'].dt.dayofweek # Monday=0, Sunday=6
        combined_df['timeofday'] = combined_df['datetime'].dt.hour
    else:
        print("Warning: 'datetime' column not found. Cannot add time-based features.")

    # Merge with machine data
    if 'machineID' in combined_df.columns and 'machineID' in machines.columns:
        combined_df = pd.merge(combined_df, machines, on='machineID', how='left')
    else:
        print("Warning: 'machineID' column not found in combined events or machines data. Cannot merge machine info.")
        # Add empty columns if merge fails but columns are expected downstream
        if 'model' not in combined_df.columns:
            combined_df['model'] = pd.NA
        if 'age' not in combined_df.columns:
            combined_df['age'] = pd.NA
    
    return combined_df

def visualize_maintenance_durations(maint_df, output_dir="plots", unit='days'):
    """
    Calculates the duration between consecutive maintenance events for each machine
    and plots a histogram of these durations.

    Parameters:
    -----------
    maint_df : pandas.DataFrame
        DataFrame containing maintenance events with 'machineID' and 'datetime'.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    unit : str, optional
        The unit for time duration ('days' or 'hours'), by default 'days'.
    """
    required_cols = ["machineID", "datetime"]
    if not all(col in maint_df.columns for col in required_cols):
        print(f"Error: Maintenance DataFrame missing required columns: {required_cols}.")
        return

    df = maint_df.copy()
    
    # Ensure datetime is sorted within each machine
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by=["machineID", "datetime"], inplace=True)

    # Calculate time difference between consecutive events per machine
    df['time_diff'] = df.groupby("machineID")['datetime'].diff()

    # Drop the first event for each machine (as it has no preceding event)
    valid_durations = df.dropna(subset=['time_diff'])

    if valid_durations.empty:
        print("Not enough consecutive maintenance events found to calculate durations.")
        return

    # Convert timedelta to the specified unit
    if unit == 'days':
        durations = valid_durations['time_diff'].dt.total_seconds() / (24 * 3600)
        x_label = f"Duration Between Maintenance Events ({unit.capitalize()})"
    elif unit == 'hours':
        durations = valid_durations['time_diff'].dt.total_seconds() / 3600
        x_label = f"Duration Between Maintenance Events ({unit.capitalize()})"
    else:
        print(f"Error: Invalid unit '{unit}'. Choose 'days' or 'hours'.")
        return
        
    print(f"\nCalculated {len(durations)} durations between maintenance events.")
    print("Duration Summary Statistics:")
    print(durations.describe())

    # Determine range for bins and ticks
    min_duration = 0 # Start bins/ticks from 0
    max_duration = math.ceil(durations.max()) # Round max duration up
    
    # Define bin edges with width 1
    bin_width = 1
    bins = np.arange(min_duration, max_duration + bin_width, bin_width)
    
    # Define tick locations with step 5
    tick_step = 5
    ticks = np.arange(min_duration, max_duration + tick_step, tick_step)

    # Plot histogram
    fig, ax = plt.subplots(figsize=(14, 7)) # Wider figure might be needed
    # Capture the counts and bin edges from hist()
    counts, bin_edges, patches = ax.hist(durations, bins=bins, edgecolor='black') 

    # Add text labels for counts above each bar
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for count, x in zip(counts, bin_centers):
        if count > 0: # Only add text for non-empty bins
            ax.text(x, count, str(int(count)), ha='center', va='bottom', fontsize=8) 

    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Time Between Consecutive Maintenance Events (All Machines)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set x-axis ticks
    ax.set_xticks(ticks)
    ax.tick_params(axis='x', rotation=45) # Rotate ticks if they overlap

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"event_maintenance_duration_histogram_{unit}.png")
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved maintenance duration histogram: {plot_filename}")

def print_intra_machine_duplicates_for_year(df_combined, machine_id):
    """
    Filters data for a specific year, then checks for and prints records
    that share the same timestamp within each individual machine.

    Parameters:
    -----------
    df_combined : pandas.DataFrame
        The combined DataFrame containing event/telemetry data with 'datetime' 
        and 'machineID' columns.
    year : int
        The year to filter the data for (e.g., 2015).
    """
    if 'datetime' not in df_combined.columns or 'machineID' not in df_combined.columns:
        print("Error: DataFrame must contain 'datetime' and 'machineID' columns.")
        return

    # Ensure datetime is in the correct format
    df = df_combined.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
            print(f"Error converting 'datetime' column to datetime objects: {e}")
            return

    df = df[df['machineID']==machine_id]

    # Filter data for the specified year
    start_date = f"2015-01-01"
    end_date = f"2015-12-31"
    print(f"\nChecking for intra-machine duplicate timestamps between {start_date} and {end_date}...")
    
    try:
        df_year = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
    except TypeError:
         print(f"Error filtering dates. Ensure 'datetime' column is comparable.")
         return

    if df_year.empty:
        print(f"No data found for the year {year}.")
        return

    found_any_duplicates = False
    grouped = df_year.groupby('machineID')

    for machine_id, machine_group in grouped:
        # keep=False marks ALL occurrences of duplicates
        duplicates = machine_group[machine_group.duplicated(subset=['datetime'], keep=False)]
        
        if not duplicates.empty:
            found_any_duplicates = True
            print(f"\n--- Duplicates found for Machine ID: {machine_id} ---")
            # Sort by datetime to see duplicates together
            print(duplicates.sort_values(by='datetime'))

    if not found_any_duplicates:
        print(f"\nNo records with duplicate timestamps found within any single machine for the year {year}.")
    else:
        print("\nFinished checking all machines.")

def plot_events_by_timeofday(df_pivoted, output_dir="plots"):
    """
    Plots the total count of event records (where maintenance, error, or failure
    flags are true) by hour of the day.

    Parameters:
    -----------
    df_pivoted : pandas.DataFrame
        The curated pivoted DataFrame with 'datetime', 'isMaintenanceEvent',
        'isErrorEvent', 'isFailureEvent'.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    required_cols = ['datetime', 'isMaintenanceEvent', 'isErrorEvent', 'isFailureEvent']
    if not all(col in df_pivoted.columns for col in required_cols):
        print(f"Error: DataFrame must contain required columns for hourly plot: {required_cols}")
        return

    df = df_pivoted.copy()
    # Ensure datetime is in the correct format
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
            print(f"Error converting 'datetime' column to datetime objects: {e}")
            return

    print("\nPlotting total event distribution by time of day...")

    # Extract hour
    df['hour'] = df['datetime'].dt.hour

    # Filter rows where at least one event flag is true
    df_events_only = df[df['isMaintenanceEvent'] | df['isErrorEvent'] | df['isFailureEvent']]

    if df_events_only.empty:
        print("No records with maintenance, error, or failure flags found.")
        return

    # Group by hour and count occurrences
    hourly_event_counts = df_events_only.groupby('hour').size()

    # Ensure all hours 0-23 are present
    hourly_event_counts = hourly_event_counts.reindex(range(24), fill_value=0)

    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Create a simple bar plot (not stacked)
    hourly_event_counts.plot(kind='bar', ax=ax, edgecolor='grey', color='#1f77b4')

    # Set labels and title
    ax.set_title('Total Event Records by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Event Records')
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, "event_total_events_by_timeofday.png")

    try:
        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"Saved total event distribution by time of day: {plot_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    finally:
        plt.close(fig)

def plot_errors_by_timeofday(df_pivoted, output_dir="plots"):
    """
    Plots the total count of error event records by hour of the day.

    Parameters:
    -----------
    df_pivoted : pandas.DataFrame
        The curated pivoted DataFrame with 'datetime' and 'isErrorEvent'.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    required_cols = ['datetime', 'isErrorEvent']
    if not all(col in df_pivoted.columns for col in required_cols):
        print(f"Error: DataFrame must contain required columns for hourly plot: {required_cols}")
        return

    df = df_pivoted.copy()
    # Ensure datetime is in the correct format
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
            print(f"Error converting 'datetime' column to datetime objects: {e}")
            return

    print("\nPlotting total error distribution by time of day...")

    # Filter for rows where isErrorEvent is True
    df_errors = df[df['isErrorEvent'] == True]

    if df_errors.empty:
        print("No error events found in the DataFrame.")
        return

    # Extract hour
    df_errors['hour'] = df_errors['datetime'].dt.hour

    # Group by hour and count occurrences
    hourly_error_counts = df_errors.groupby('hour').size()

    # Ensure all hours 0-23 are present
    hourly_error_counts = hourly_error_counts.reindex(range(24), fill_value=0)

    # Create a simple bar plot
    fig, ax = plt.subplots(figsize=(14, 7))

    hourly_error_counts.plot(kind='bar', ax=ax, edgecolor='grey', color='#d62728') # Use red color

    # Set labels and title
    ax.set_title('Total Error Records by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Error Records')
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, "event_total_errors_by_timeofday.png")

    try:
        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"Saved total error distribution by time of day: {plot_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    finally:
        plt.close(fig)

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

def filter_datetime(df):
    start_date = f"2015-01-01"
    end_date = f"2016-01-01"
    return df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

def visualize_curated_events(curated_events, machine_id, output_dir="plots"):
    """
    Plots a stacked histogram of days since the last *maintenance* event for a
    specific machine, color-coded by the eventType of the *current* event.

    Parameters:
    -----------
    curated_events : pandas.DataFrame
        DataFrame output from curate_events, containing 'datetime', 'machineID',
        'eventType', 'daysSinceLastMaintenanceEvent'.
    machine_id : int
        The ID of the machine to visualize.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    # Updated required columns - no longer need is_6am for this plot
    required_cols = ['datetime', 'machineID', 'eventType', 'daysSinceLastMaintenanceEvent']
    if not all(col in curated_events.columns for col in required_cols):
        print(f"Error: Input DataFrame missing one or more required columns: {required_cols}")
        return

    df = curated_events.copy()

    # Filter for the specific machine
    df_machine = df[df['machineID'] == machine_id]

    if df_machine.empty:
        print(f"No curated event data found for machine {machine_id}.")
        return

    print(f"\nPlotting histogram of days since last maintenance event for machine {machine_id}, colored by event type...")

    # Define colors for event types (ensure consistency with other plots if applicable)
    event_type_colors = {
        'error': '#d62728',      # Red
        'failure': '#1f77b4',     # Blue
        'maintenance': '#2ca02c'   # Green
    }
    # Order categories for consistent legend
    event_type_order = ['error', 'failure', 'maintenance']
    # --- Create Stacked Histogram ---
    fig, ax = plt.subplots(figsize=(14, 7))

    # Data to plot: list of arrays, one for each category
    data_to_plot = []
    labels = []
    colors = []
    for etype in event_type_order:
        # Select data for the current event type for the specific machine
        etype_data = df_machine.loc[df_machine['eventType'] == etype, 'daysSinceLastMaintenanceEvent']
        # Drop NaNs as they cannot be plotted in histogram (events before first maint)
        etype_data = etype_data.dropna()
        # Only include if there's data for this type
        if not etype_data.empty:
            data_to_plot.append(etype_data)
            labels.append(etype)
            colors.append(event_type_colors.get(etype, '#7f7f7f')) # Use grey as fallback

    if not data_to_plot:
        print(f"No event data to plot for machine {machine_id} after grouping by type.")
        plt.close(fig)
        return
        
    # Determine bins with bin_width = 1
    all_plot_data = pd.concat(data_to_plot)
    max_days = all_plot_data.max()
    min_days = 0
    bin_width = 1 # Set bin width to 1 day
    if not np.isfinite(max_days):
        max_days = 100 
        print(f"Warning: Could not determine max days, using default {max_days}.")

    # Create bins centered around integers for width=1
    bins = np.arange(min_days - bin_width/2, max_days + bin_width, bin_width) 

    # Plot the stacked histogram
    ax.hist(data_to_plot, bins=bins, stacked=True, label=labels, color=colors, edgecolor='black')

    # Set labels and title
    ax.set_title(f'Days Since Last Maintenance Event for Machine ID {machine_id} (Colored by Event Type)')
    ax.set_xlabel('Days Since Previous Maintenance Event')
    ax.set_ylabel('Count of Events')
    ax.legend(title='Event Type') # Updated legend title
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set x-axis ticks explicitly, e.g., every 5 days
    tick_interval = 5
    ax.set_xticks(np.arange(min_days, max_days + tick_interval, tick_interval))
    # Ensure 0 is included if relevant and within the range
    current_xticks = ax.get_xticks()
    if min_days <= 0 < max_days and 0 not in current_xticks:
        ax.set_xticks(np.sort(np.append(ax.get_xticks(), 0)))
        
    ax.tick_params(axis='x', rotation=45)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Update filename to reflect the plot content
    plot_filename = os.path.join(output_dir, f"event_days_since_last_maint_by_type_machine_{machine_id}.png")

    try:
        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"Saved days since last maintenance event histogram (colored by type): {plot_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    finally:
        plt.close(fig)

def visualize_daily_event_histogram(curated_events, machine_id=None, output_dir="plots"):
    """
    Plots a histogram of event counts per day, stacked and colored by curated_eventtype.
    Can optionally filter for a specific machine ID.

    Parameters:
    -----------
    curated_events : pandas.DataFrame
        DataFrame output from curate_events, containing at least 'datetime',
        'machineID', and 'curated_eventtype'.
    machine_id : int, optional
        If provided, filters the data for this specific machine ID. 
        If None (default), plots data for all machines.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    required_cols = ['datetime', 'machineID', 'curated_eventtype']
    if not all(col in curated_events.columns for col in required_cols):
        print(f"Error: Input DataFrame missing required columns for daily histogram: {required_cols}")
        return

    df_plot = curated_events.copy()
    plot_title = "Daily Event Counts by Type (All Machines)"
    filename_suffix = "all_machines"

    if machine_id is not None:
        df_plot = df_plot[df_plot['machineID'] == machine_id]
        plot_title = f"Daily Event Counts by Type (Machine ID {machine_id})"
        filename_suffix = f"machine_{machine_id}"

    if df_plot.empty:
        print(f"No event data found for the specified scope (Machine ID: {machine_id}) to plot daily histogram.")
        return

    # Ensure datetime is in the correct format
    if not pd.api.types.is_datetime64_any_dtype(df_plot['datetime']):
        try:
            df_plot['datetime'] = pd.to_datetime(df_plot['datetime'])
        except Exception as e:
            print(f"Error converting 'datetime' column to datetime objects: {e}")
            return

    print(f"\nPlotting daily event histogram for Machine ID: {machine_id if machine_id else 'All'}...")

    # Group by day and the new curated_eventtype
    daily_counts = df_plot.groupby([pd.Grouper(key='datetime', freq='D'), 'curated_eventtype']).size()

    # Unstack eventType to columns
    daily_counts_unstacked = daily_counts.unstack(level='curated_eventtype', fill_value=0)

    # Ensure all days in the range are present
    if not daily_counts_unstacked.empty:
        min_date = daily_counts_unstacked.index.min()
        max_date = daily_counts_unstacked.index.max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        daily_counts_unstacked = daily_counts_unstacked.reindex(full_date_range, fill_value=0)
    else:
        print("No data to plot after grouping.")
        return

    # Define colors for the new 5 curated types
    curated_event_type_colors = {
        'error':                 '#d62728', # Red
        'failure (scheduled)':   '#8c564b', # Brown
        'failure (unscheduled)': '#ff7f0e', # Orange
        'maintenance (scheduled)': '#2ca02c', # Dark Green
        'maintenance (unscheduled)': '#98df8a'  # Light Green
    }
    default_color = '#7f7f7f' # Grey for unknown/other combinations

    # Get colors in the order of columns present
    plot_colors = [curated_event_type_colors.get(col, default_color) for col in daily_counts_unstacked.columns]

    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(17, 5))
    daily_counts_unstacked.plot(kind='bar', stacked=True, ax=ax, color=plot_colors, width=1.0)

    # --- Finalize Plot ---
    ax.set_title(plot_title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Events")
    ax.legend(title='Curated Event Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    # Improve date formatting on x-axis
    # Show fewer ticks for daily data over a long period
    tick_frequency = max(1, len(daily_counts_unstacked) // 30) # Aim for ~30 ticks
    ax.set_xticks(ax.get_xticks()[::tick_frequency])
    fig.autofmt_xdate()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"event_daily_histogram_curated_{filename_suffix}.png")

    try:
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout for legend
        plt.savefig(plot_filename)
        print(f"Saved daily event histogram plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving daily event histogram plot: {e}")
    finally:
        plt.close(fig)

def pivot_events_by_category(df):
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

def curate_pivoted_events(pivoted_df):
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
    required_cols = ['machineID', 'datetime']
    if not all(col in pivoted_df.columns for col in required_cols):
        print("Error: Pivoted DataFrame missing 'machineID' or 'datetime'. Cannot curate.")
        return pivoted_df

    df = pivoted_df.copy()

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

    print(f"Curated pivoted DataFrame shape: {df.shape}")
    return df

def plot_curated_eventtype_counts(curated_pivoted_df, machine_id, output_dir="plots"):
    """
    Plots a horizontal stacked bar chart showing counts of CuratedEventType,
    colored by the combination of 6am status and scheduled status, for a specific machine.

    Parameters:
    -----------
    curated_pivoted_df : pandas.DataFrame
        DataFrame output from curate_pivoted_events, containing columns
        'machineID', 'CuratedEventType', 'is_6am', 'isScheduled'.
    machine_id : int
        The ID of the machine to plot.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    required_cols = ['machineID', 'CuratedEventType', 'is_6am', 'isScheduled']
    if not all(col in curated_pivoted_df.columns for col in required_cols):
        print(f"Error: Input DataFrame missing one or more required columns for CuratedEventType plot: {required_cols}")
        return

    df_machine = curated_pivoted_df[curated_pivoted_df['machineID'] == machine_id].copy()

    if df_machine.empty:
        print(f"No curated pivoted data found for machine {machine_id} to plot CuratedEventType counts.")
        return

    print(f"\nPlotting CuratedEventType counts for machine {machine_id}...")

    # Create combined status column
    def get_status_category(row):
        time_part = "6am" if row['is_6am'] else "Other"
        sched_part = "Sched" if row['isScheduled'] else "Unsched"
        return f"{time_part}-{sched_part}"

    df_machine['status_category'] = df_machine.apply(get_status_category, axis=1)

    # Group by CuratedEventType and the new status category
    counts = df_machine.groupby(['CuratedEventType', 'status_category']).size().unstack(fill_value=0)

    # Define order and ensure all categories are present
    status_order = ['6am-Sched', '6am-Unsched', 'Other-Sched', 'Other-Unsched']
    counts = counts.reindex(columns=status_order, fill_value=0)

    # Filter out rows (CuratedEventTypes) where all counts are zero
    counts = counts[counts.sum(axis=1) > 0]

    if counts.empty:
        print(f"No non-zero CuratedEventType counts found for machine {machine_id} after grouping.")
        return

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted size for potentially more categories

    # Define colors for the 4 status categories
    colors = {
        '6am-Sched': '#1f77b4', # Blue
        '6am-Unsched': '#aec7e8', # Light Blue
        'Other-Sched': '#2ca02c', # Green
        'Other-Unsched': '#98df8a'  # Light Green
    }
    plot_colors = [colors[col] for col in counts.columns]

    # Plot stacked horizontal bar chart
    counts.plot(kind='barh', stacked=True, ax=ax, color=plot_colors)

    # Add labels and title
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Curated Event Type")
    ax.set_title(f"Curated Event Type Counts by Time/Schedule (Machine ID {machine_id})")
    ax.legend(title='Time & Schedule Status', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Add value labels (optional, can get crowded)
    # Consider adding only if counts are significant or total

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"curated_eventtype_counts_machine_{machine_id}.png")

    try:
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plt.savefig(plot_filename)
        print(f"Saved curated event type counts plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving curated event type counts plot: {e}")
    finally:
        plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Define base path for data
    data_path = "telemetry"
    
    # Define file paths
    errors_file = os.path.join(data_path, "PdM_errors.csv")
    failures_file = os.path.join(data_path, "PdM_failures.csv")
    maintenance_file = os.path.join(data_path, "PdM_maint.csv")
    machines_file = os.path.join(data_path, "PdM_machines.csv") 
    telemetry_file = os.path.join(data_path, "PdM_telemetry.csv")

    # --- Load Data --- 
    try:
        PDM_Errors = pd.read_csv(errors_file, parse_dates=['datetime'])
        Failures = pd.read_csv(failures_file, parse_dates=['datetime'])
        Maintenance = pd.read_csv(maintenance_file, parse_dates=['datetime'])
        PDM_Machines = pd.read_csv(machines_file) 
        PDM_Telemetry = pd.read_csv(telemetry_file, parse_dates=['datetime']) # Load telemetry
        
        print("Data loaded successfully.")
        
        # --- Process Event Data --- 
        combined_events = process_event_data(PDM_Errors, Failures, Maintenance, PDM_Machines)
        print_intra_machine_duplicates_for_year(combined_events, machine_id=1)

        # Pivot the combined events
        filtered_events = filter_datetime(combined_events)
        pivoted_combined_events = pivot_events_by_category(filtered_events)
        print("\nPivoted Combined Events DataFrame head:")
        print(pivoted_combined_events.head(50))

        # Curate the pivoted events
        curated_pivoted = curate_pivoted_events(pivoted_combined_events)
        print("\nCurated Pivoted Events DataFrame head:")
        print(curated_pivoted.head(50))

        # Plot curated event type counts
        plot_curated_eventtype_counts(curated_pivoted, machine_id=1, output_dir='plots')

        plot_events_by_timeofday(curated_pivoted, output_dir='plots')
        plot_errors_by_timeofday(curated_pivoted, output_dir='plots')
        # visualize_maintenance_durations(Maintenance, unit='days') # Calculate in days - Uses original Maintenance data

        # print("\nCombined and Enriched Events DataFrame head:")
        # print(combined_events.head())
        # print("\nColumns:", combined_events.columns)

        # --- Visualize Data --- 
        # Event Timeseries visualization
        # visualize_timeseries(combined_events) 
   


    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the CSV files are in the '{data_path}' directory.")
    except KeyError as e:
        print(f"Error processing/visualizing data: Missing expected column {e}. Check CSV file structures and processing steps.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # # Example with dummy data for demonstration (Keep commented out or remove)
    # PDM_Errors = pd.DataFrame({
    #     'errorID': ['E1', 'E2', 'E3'],
    #     'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
    #     'machineID': [1, 2, 1]
    # })
    # Failures = pd.DataFrame({
    #     'failure': ['F1', 'F2', 'F3'],
    #     'datetime': pd.to_datetime(['2023-02-01', '2023-02-02', '2023-02-03']),
    #     'machineID': [3, 1, 2]
    # })
    # Maintenance = pd.DataFrame({
    #     'comp': ['M1', 'M2', 'M3'],
    #     'datetime': pd.to_datetime(['2023-03-01', '2023-03-02', '2023-03-03']),
    #     'machineID': [1, 3, 1]
    # })
    # 
    # # Process the dataframes
    # combined_events = process_event_data(PDM_Errors, Failures, Maintenance)
    # 
    # # Display the result
    # print(combined_events) 
    # # Visualize
    # if 'machineID' in combined_events.columns and 'datetime' in combined_events.columns:
    #     visualize_timeseries(combined_events) 
