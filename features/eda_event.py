import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from scipy import signal # Import scipy.signal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose
import pywt

def plot_maintenance_durations(maint_df, output_dir="plots", unit='days'):
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
    Plots the distribution of event records by hour of the day, stacked and colored
    by the combination of specific errors (error1-5) present.

    Parameters:
    -----------
    df_pivoted : pandas.DataFrame
        The curated pivoted DataFrame with 'datetime' and error columns
        ('error1'-'error5').
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    required_base = ['datetime']
    error_cols = [f'error{i}' for i in range(1, 6)]
    # Find which error columns actually exist in the dataframe
    error_cols_present = [col for col in error_cols if col in df_pivoted.columns]

    if not error_cols_present:
        print("Error: No specific error columns (error1-5) found for grouping.")
        return

    required_cols = required_base + error_cols_present
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

    print("\nPlotting event distribution by specific error combination by time of day...")

    # Extract hour
    df['hour'] = df['datetime'].dt.hour

    # Create list_of_errors column
    def get_error_list(row):
        errors = [col for col in error_cols_present if row[col] != '']
        return ",".join(sorted(errors)) if errors else "No Error"

    df['list_of_errors'] = df.apply(get_error_list, axis=1)

    # Group by hour and list_of_errors, count occurrences
    hourly_counts = df.groupby(['hour', 'list_of_errors']).size().unstack(fill_value=0)

    # Ensure all hours 0-23 are present
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0)

    # Filter out error combinations that have zero counts across all hours
    hourly_counts = hourly_counts.loc[:, (hourly_counts.sum(axis=0) > 0)]

    if hourly_counts.empty:
        print("No non-zero counts found after grouping by error combination and hour.")
        return

    # Define colors for error combinations (using a colormap for potentially many combos)
    unique_error_combos = sorted(hourly_counts.columns)
    # Ensure "No Error" is plotted first/distinctly if present
    if "No Error" in unique_error_combos:
        unique_error_combos.remove("No Error")
        unique_error_combos.insert(0, "No Error")
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_error_combos)))
    error_combo_color_map = dict(zip(unique_error_combos, colors))
    # Override color for "No Error" to be less prominent
    if "No Error" in error_combo_color_map:
        error_combo_color_map["No Error"] = '#cccccc' # Light grey

    plot_colors = [error_combo_color_map.get(col) for col in hourly_counts.columns]

    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 7))

    hourly_counts.plot(kind='bar', stacked=True, ax=ax,
                       color=plot_colors, edgecolor='grey')

    # Set labels and title
    ax.set_title('Event Distribution by Specific Error Combination by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Events')
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    # Limit legend entries if too many combinations
    max_legend_entries = 20
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > max_legend_entries:
        # Display only the top N most frequent combinations in the legend
        top_combos = hourly_counts.sum().nlargest(max_legend_entries).index
        handles_labels_to_show = {label: handle for handle, label in zip(handles, labels) if label in top_combos}
        ax.legend(handles_labels_to_show.values(), handles_labels_to_show.keys(), title=f'Error Combinations (Top {max_legend_entries})', bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax.legend(title='Error Combination', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, "event_by_error_combo_by_timeofday.png") # New filename

    try:
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Make space for legend
        plt.savefig(plot_filename)
        print(f"Saved event distribution by error combination by time of day: {plot_filename}")
    except Exception as e:
        print(f"\nError saving plot: {e}")
    finally:
        plt.close(fig)

def plot_errors_by_timeofday(df_pivoted, output_dir="plots"):
    """
    Plots the distribution of errors by hour of the day, stacked and colored
    by CuratedEventType.

    Parameters:
    -----------
    df_pivoted : pandas.DataFrame
        The curated pivoted DataFrame with 'datetime', 'isErrorEvent', 
        and 'CuratedEventType'.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    required_cols = ['datetime', 'isErrorEvent', 'CuratedEventType']
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

    print(f"\nPlotting error distribution by CuratedEventType by time of day...")

    # Filter for rows where isErrorEvent is True
    df_errors = df[df['isErrorEvent'] == True]

    if df_errors.empty:
        print("No error events found in the DataFrame.")
        return

    # Extract hour
    df_errors['hour'] = df_errors['datetime'].dt.hour

    # Group by hour and CuratedEventType, count occurrences
    hourly_counts = df_errors.groupby(['hour', 'CuratedEventType']).size().unstack(fill_value=0)

    # Ensure all hours 0-23 are present
    hourly_counts = hourly_counts.reindex(range(24), fill_value=0)

    # Filter out CuratedEventTypes that have zero counts across all hours
    hourly_counts = hourly_counts.loc[:, (hourly_counts.sum(axis=0) > 0)]

    if hourly_counts.empty:
        print("No non-zero counts found after grouping errors by CuratedEventType and hour.")
        return

    # Define colors for CuratedEventType values (ensure consistency if possible)
    curated_colors = {
        'Maintenance': '#2ca02c',
        'Error': '#d62728',
        'Failure': '#ff7f0e',
        'Maintenance-Error': '#9467bd',
        'Maintenance-Failure': '#8c564b',
        'Error-Failure': '#e377c2',
        'Maintenance-Error-Failure': '#7f7f7f' # Grey
    }
    default_color = '#bcbd22' # Olive
    plot_colors = [curated_colors.get(col, default_color) for col in hourly_counts.columns]

    # Create the stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 7))

    hourly_counts.plot(kind='bar', stacked=True,
                      color=plot_colors,
                      ax=ax, edgecolor='grey')

    # Set labels and title
    ax.set_title('Distribution of Errors by Curated Event Type by Hour of Day')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Error Events') # Changed label
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.legend(title='Curated Event Type', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, "event_errors_by_curatedtype_by_timeofday.png") # New filename

    try:
        # Adjust layout to prevent legend overlap
        plt.tight_layout(rect=[0, 0, 0.88, 1]) # Make space for legend
        plt.savefig(plot_filename)
        print(f"Saved error distribution by curated type by time of day: {plot_filename}")
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

def plot_daily_event_histogram(curated_pivoted_df, machine_id=None, output_dir="plots"):
    """
    Plots a histogram of event counts per day, stacked and colored by CuratedEventType.
    Can optionally filter for a specific machine ID.

    Parameters:
    -----------
    curated_pivoted_df : pandas.DataFrame
        DataFrame output from curate_pivoted_events, containing at least 'datetime',
        'machineID', and 'CuratedEventType'.
    machine_id : int, optional
        If provided, filters the data for this specific machine ID. 
        If None (default), plots data for all machines.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    required_cols = ['datetime', 'machineID', 'CuratedEventType']
    if not all(col in curated_pivoted_df.columns for col in required_cols):
        print(f"Error: Input DataFrame missing one or more required columns for daily histogram: {required_cols}")
        return

    df_plot = curated_pivoted_df.copy()
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

    # Group by day and the new CuratedEventType
    daily_counts = df_plot.groupby([pd.Grouper(key='datetime', freq='D'), 'CuratedEventType']).size()

    # Unstack eventType to columns
    daily_counts_unstacked = daily_counts.unstack(level='CuratedEventType', fill_value=0)

    # Ensure all days in the range are present
    if not daily_counts_unstacked.empty:
        min_date = daily_counts_unstacked.index.min()
        max_date = daily_counts_unstacked.index.max()
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        daily_counts_unstacked = daily_counts_unstacked.reindex(full_date_range, fill_value=0)
    else:
        print("No data to plot after grouping.")
        return

    # Define colors for CuratedEventType values
    curated_event_type_colors = {
        'Error': '#d62728',
        'Maintenance': '#2ca02c',
        'Failure': '#ff7f0e',
        'Maintenance-Error': '#9467bd',
        'Maintenance-Failure': '#8c564b',
        'Error-Failure': '#e377c2',
        'Maintenance-Error-Failure': '#7f7f7f'
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
    ax.grid(axis='y', linestyle='--', alpha=0.7)

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
    cols_to_impute = [col for col in df.columns if col.startswith('CountOfError') and col.endswith('SinceLastMaintenance')]

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


def track_error_history(curated_pivoted_df):
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

def plot_maintenance_by_recent_errors(curated_pivoted_df, machine_id=None, output_dir="plots"):
    """
    Plots a grid of bar charts comparing maintenance events for each component (1-4)
    preceded by specific errors (1-5) vs. those not, for a given machine.
    Color within bars indicates if the maintenance event was a failure.

    Parameters:
    -----------
    curated_pivoted_df : pandas.DataFrame
        DataFrame output from track_error_history, containing 'machineID',
        'isMaintenanceEvent', 'comp1'- 'comp4', and 'CountOfErrorXSinceLastMaintenance' cols.
    machine_id : int, optional
        The ID of the machine to plot. If None, plots for all machines.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    # Determine scope: single machine or all
    if machine_id is not None:
        df_scope = curated_pivoted_df[curated_pivoted_df['machineID'] == machine_id].copy()
        scope_label = f"Machine ID: {machine_id}"
        filename_suffix = f"machine_{machine_id}"
        if df_scope.empty:
            print(f"No data found for machine {machine_id}.")
            return
    else:
        df_scope = curated_pivoted_df.copy()
        scope_label = "All Machines"
        filename_suffix = "all_machines"

    base_required = ['machineID', 'isMaintenanceEvent']
    if not all(col in df_scope.columns for col in base_required):
        print(f"Error: Input DataFrame missing base required columns: {base_required}")
        return

    components = [f'comp{i}' for i in range(1, 5)]
    if not all(comp in df_scope.columns for comp in components):
        missing_comps = [c for c in components if c not in df_scope.columns]
        print(f"Error: Input DataFrame missing required component columns: {missing_comps}")
        return

    print(f"\nPlotting component maintenance preceded by errors (including double error 2&3) for {scope_label}...")

    # Prepare subplots (4 rows for components, 6 columns for errors 1-5 + double error)
    fig, axes = plt.subplots(4, 6, figsize=(18, 12), sharey=True, sharex=True) # Changed figsize and grid size

    fig.suptitle(f"Maintenance Events Preceded by Errors ({scope_label})", y=1.0)

    # Check if the double error count column exists
    double_error_count_col = 'CountOfDoubleErrorsSinceLastMaintenance'
    has_double_error_col = double_error_count_col in df_scope.columns
    if not has_double_error_col:
        print(f"Warning: Column '{double_error_count_col}' not found. Skipping double error plot.")

    for comp_idx, comp_name in enumerate(components):
        # Filter for maintenance events involving the current component for the current scope
        df_maint_comp = df_scope[(df_scope['isMaintenanceEvent'] == True) & (df_scope[comp_name] != '')].copy()

        if df_maint_comp.empty:
            print(f"No maintenance events involving {comp_name} found for {scope_label}. Skipping row.")
            # Optionally hide the row or display a message
            for plot_col_idx in range(6): # Updated range to 6
                axes[comp_idx, plot_col_idx].set_visible(False)
            axes[comp_idx, 0].set_ylabel(f"{comp_name}\n(No Data)", rotation=0, labelpad=40, ha='right', va='center')
            continue

        # Create temporary boolean columns for this component's maintenance subset
        error_check_cols_present = {}
        # Individual errors 1-5
        for i in range(1, 6):
            count_col = f'CountOfError{i}SinceLastMaintenance'
            has_error_col = f'_hasError{i}Occured'
            if count_col in df_maint_comp.columns:
                df_maint_comp[has_error_col] = (df_maint_comp[count_col] > 0).fillna(False)
                error_check_cols_present[i] = has_error_col
            else:
                error_check_cols_present[i] = None
        # Double error
        has_double_error_check_col = f'_hasDoubleErrorOccured'
        if has_double_error_col:
            df_maint_comp[has_double_error_check_col] = (df_maint_comp[double_error_count_col] > 0).fillna(False)
            error_check_cols_present['double'] = has_double_error_check_col
        else:
             error_check_cols_present['double'] = None

        # Loop through error types (1-5) and the double error
        for plot_col_idx, error_key in enumerate(list(range(1, 6)) + ['double']):
            ax = axes[comp_idx, plot_col_idx]
            check_col = error_check_cols_present.get(error_key)
            
            if check_col:
                # Group by whether the error occurred AND failure status
                counts = df_maint_comp.groupby([check_col, 'isFailureEvent']).size()
                # Unstack to get Failure status as columns
                counts_unstacked = counts.unstack(level='isFailureEvent', fill_value=0)
                # Ensure both error statuses (True/False) and failure statuses (True/False) are present
                counts_unstacked = counts_unstacked.reindex(index=[True, False], fill_value=0)
                counts_unstacked = counts_unstacked.reindex(columns=[True, False], fill_value=0)
                # Rename columns for clarity in legend
                counts_unstacked.columns = ['Failure', 'No Failure']

                # Plot stacked bar chart
                counts_unstacked[['Failure', 'No Failure']].plot(kind='bar', stacked=True, ax=ax, rot=0,
                                                                color=['#ff7f0e', '#1f77b4'])

                # Set title based on error type
                if isinstance(error_key, int):
                    ax.set_title(f"Error {error_key} Occurred?")
                else: # Double error
                    ax.set_title(f"Dbl Err(2&3) Occurred?")
                ax.set_xlabel("") # Keep x-label minimal
                ax.set_xticklabels(['True', 'False'])
                if plot_col_idx == 0:
                    # Add component name to Y label
                    ax.set_ylabel(f"{comp_name}\nMaint Events", rotation=0, labelpad=40, ha='right', va='center')
                # Add legend only once for the whole figure
                if comp_idx == 0 and plot_col_idx == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, title='Failure Status', bbox_to_anchor=(1.0, 0.9), loc='upper left')
                ax.get_legend().remove() # Remove individual subplot legends
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                # Handle case where CountOfErrorX or DoubleError column was missing
                if isinstance(error_key, int):
                     ax.set_title(f"Error {error_key}\n(No Data)")
                else:
                     ax.set_title(f"Dbl Err(2&3)\n(No Data)")
                ax.set_xticklabels([])
                ax.set_yticks([])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Update filename slightly (optional)
    plot_filename = os.path.join(output_dir, f"event_maint_by_comp_preceded_by_errors_incl_double_{filename_suffix}.png")

    try:
        plt.tight_layout(rect=[0.05, 0, 0.9, 0.97]) # Adjust layout further for figure legend
        plt.savefig(plot_filename)
        print(f"Saved maintenance preceded by errors plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving maintenance preceded by errors plot: {e}")
    finally:
        plt.close(fig)

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
    plot_filename = os.path.join(output_dir, f"event_curated_eventtype_counts_machine_{machine_id}.png")

    try:
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
        plt.savefig(plot_filename)
        print(f"Saved curated event type counts plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving curated event type counts plot: {e}")
    finally:
        plt.close(fig)

def plot_unplanned_failure_components(curated_pivoted_df, output_dir="plots"):
    """
    Plots a bar chart showing the frequency of different component combinations 
    involved in unplanned failure events.

    Parameters:
    -----------
    curated_pivoted_df : pandas.DataFrame
        DataFrame output from curate_pivoted_events, containing 'isFailureEvent',
        'isScheduled', and component columns ('comp1'-'comp4').
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    base_required = ['isFailureEvent', 'isScheduled']
    components = [f'comp{i}' for i in range(1, 5)]
    required_cols = base_required + components

    if not all(col in curated_pivoted_df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in curated_pivoted_df.columns]
        print(f"Error: Input DataFrame missing required columns for unplanned failure plot: {missing}")
        return

    df = curated_pivoted_df.copy()

    # Filter for unplanned failures
    df_unplanned_failures = df[(df['isFailureEvent'] == True) & (df['isScheduled'] == False)].copy()

    if df_unplanned_failures.empty:
        print("No unplanned failure events found in the data.")
        return

    print(f"\nPlotting unplanned failure component combinations...")

    # Function to get the list of non-empty component columns for a row
    def get_failed_comps(row):
        failed = [comp for comp in components if row[comp] != '']
        return ",".join(sorted(failed)) # Sort for consistency

    # Create the new column
    df_unplanned_failures['list_of_failure_components'] = df_unplanned_failures.apply(get_failed_comps, axis=1)

    # Calculate counts for each combination
    component_combo_counts = df_unplanned_failures['list_of_failure_components'].value_counts()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 7))

    component_combo_counts.plot(kind='bar', ax=ax, rot=45, color='#ff7f0e') # Orange color

    # Add labels and title
    ax.set_title("Frequency of Component Combinations in Unplanned Failures")
    ax.set_xlabel("Failed Component Combination")
    ax.set_ylabel("Number of Unplanned Failure Events")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add count labels on top of bars
    for container in ax.containers:
        ax.bar_label(container)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"event_unplanned_failure_component_combos.png")

    try:
        plt.tight_layout()
        plt.savefig(plot_filename)
        print(f"Saved unplanned failure component combination plot: {plot_filename}")
    except Exception as e:
        print(f"\nError saving unplanned failure plot: {e}")
    finally:
        plt.close(fig)

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

# Example usage
if __name__ == "__main__":
    # Define base path for data
    data_path = "telemetry"
    
    # Define output directory for CSVs
    output_csv_dir = "data"
    os.makedirs(output_csv_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_csv_dir}")

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

    plot_maintenance_durations(Maintenance, unit='days') # Calculate in days - Uses original Maintenance data

    
    # --- Process Event Data --- 
    combined_events = process_event_data(PDM_Errors, Failures, Maintenance, PDM_Machines)
    print_intra_machine_duplicates_for_year(combined_events, machine_id=1)

    # Pivot the combined events
    filtered_events = filter_datetime(combined_events)
    pivoted_combined_events = pivot_events_by_category(filtered_events)
    pivoted_combined_events.to_csv(os.path.join(output_csv_dir, "pivoted_combined_events.csv"), index=False)

    # Curate the pivoted events
    curated_pivoted = curate_pivoted_events(pivoted_combined_events)
    curated_pivoted.to_csv(os.path.join(output_csv_dir, "curated_pivoted.csv"), index=False)

    # EDA on curated pivoted
    plot_curated_eventtype_counts(curated_pivoted, machine_id=1, output_dir='plots')
    plot_events_by_timeofday(curated_pivoted, output_dir='plots')
    plot_errors_by_timeofday(curated_pivoted, output_dir='plots')
    plot_daily_event_histogram(curated_pivoted_df=curated_pivoted, machine_id=1)

    # Track error history
    curated_pivoted_with_history = track_error_history(curated_pivoted)
    curated_pivoted_with_history.to_csv(os.path.join(output_csv_dir, "curated_pivoted_with_history.csv"), index=False)

    plot_unplanned_failure_components(curated_pivoted_with_history, output_dir='plots')
    plot_maintenance_by_recent_errors(curated_pivoted_with_history, machine_id=None, output_dir='plots')

    # lagged_telemetry_df = add_basic_lag_stats(PDM_Telemetry)
    # lagged_telemetry_df.to_csv(os.path.join(output_csv_dir, "lagged_telemetry_df.csv"), index=False)

    # joined_df = join_events_and_telemetry(curated_pivoted_with_history, lagged_telemetry_df)
    # #joined_df.to_csv("joined_df.csv", index=False)

    # # Impute missing error counts
    # imputed_joined_df = impute_error_counts(joined_df)
    # imputed_joined_df.to_csv(os.path.join(output_csv_dir, "imputed_joined_df.csv"), index=False)

   
