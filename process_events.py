import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from scipy import signal # Import scipy.signal

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

def visualize_timeseries(df, output_dir="plots"):
    """
    Visualizes event timeseries for machines, grouped by model,
    plotting 10 machines per figure.
    Color represents eventType, Shape represents eventcategory.

    Parameters:
    -----------
    df : pandas.DataFrame
        Combined DataFrame with 'datetime', 'machineID', 'model', 'eventType',
        and 'eventcategory'. Assumes 'model' column exists.
    output_dir : str, optional
        Directory to save the plots, by default "plots"
    """
    if 'model' not in df.columns:
        print("Error: 'model' column not found in DataFrame. Cannot group by model.")
        return
    if df['model'].isnull().any():
        print("Warning: DataFrame contains null values in 'model' column. Machines with null models will be skipped.")
        df = df.dropna(subset=['model'])
        
    # Ensure the datetime column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['eventcategory'] = df['eventcategory'].astype(str) # Ensure category is string for mapping

    # Get unique overall event types and categories for consistent legends
    unique_event_types = sorted(df['eventType'].unique())
    unique_event_categories = sorted(df['eventcategory'].unique())

    # --- Define Color and Marker Maps (globally for consistency across plots) ---
    type_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_event_types)))
    event_type_color_map = dict(zip(unique_event_types, type_colors))

    available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H', '+', 'x']
    if len(unique_event_categories) > len(available_markers):
        print(f"Warning: More event categories ({len(unique_event_categories)}) than available markers ({len(available_markers)}). Markers will repeat.")
        markers_to_use = available_markers * (len(unique_event_categories) // len(available_markers)) + available_markers[:len(unique_event_categories) % len(available_markers)]
    else:
        markers_to_use = available_markers[:len(unique_event_categories)]
    event_category_marker_map = dict(zip(unique_event_categories, markers_to_use))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    machines_per_plot = 10
    
    # --- Group by Model and Plot --- 
    grouped_by_model = df.groupby('model')
    
    for model_name, model_group_df in grouped_by_model:
        print(f"\nProcessing Model: {model_name}")
        # Get sorted machine IDs for this specific model
        model_machines = sorted(model_group_df['machineID'].unique())
        num_model_machines = len(model_machines)
        
        if num_model_machines == 0:
            print(f"Skipping model {model_name} as it has no associated machine data after filtering.")
            continue
            
        num_plots_for_model = math.ceil(num_model_machines / machines_per_plot)
        
        print(f"Model {model_name}: {num_model_machines} machines, generating {num_plots_for_model} plots.")
        
        for i in range(num_plots_for_model):
            # Calculate indices relative to the *model* group
            start_model_index = i * machines_per_plot
            end_model_index = min((i + 1) * machines_per_plot, num_model_machines)
            
            # Get the actual machine IDs for this plot
            current_plot_machine_ids = model_machines[start_model_index:end_model_index]
            
            # Filter the model's data for the machines in the current plot
            subset_df = model_group_df[model_group_df['machineID'].isin(current_plot_machine_ids)]
            
            if subset_df.empty:
                print(f"Skipping plot for model {model_name}, machines index {start_model_index+1}-{end_model_index} (IDs: {current_plot_machine_ids}) as there is no event data.")
                continue

            fig, ax = plt.subplots(figsize=(15, 10))

            # --- Plotting (same logic as before, using subset_df) ---
            plotted_combinations = set()
            for etype in unique_event_types:
                for ecategory in unique_event_categories:
                    event_subset = subset_df[(subset_df['eventType'] == etype) & (subset_df['eventcategory'] == ecategory)]
                    if not event_subset.empty:
                        color = event_type_color_map[etype]
                        marker = event_category_marker_map[ecategory]
                        ax.scatter(event_subset['datetime'], event_subset['machineID'],
                                   label=f"{etype}_{ecategory}",
                                   color=color, marker=marker, s=60, alpha=0.8)
                        plotted_combinations.add((etype, ecategory))
            
            # --- Create Custom Legend (same logic as before) ---
            from matplotlib.lines import Line2D
            color_legend_elements = [Line2D([0], [0], marker='o', color='w', label=etype,
                                      markerfacecolor=event_type_color_map[etype], markersize=10)
                                     for etype in unique_event_types]
            marker_legend_elements = [Line2D([0], [0], marker=event_category_marker_map[ecategory], color='w', label=ecategory,
                                       markerfacecolor='grey', markeredgecolor='grey', markersize=10)
                                      for ecategory in unique_event_categories]
            leg1 = ax.legend(handles=color_legend_elements, title="Event Type (Color)", 
                             bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.add_artist(leg1)
            leg2 = ax.legend(handles=marker_legend_elements, title="Event Category (Shape)", 
                             bbox_to_anchor=(1.05, 0.5), loc='center left')

            # --- Axis labels, title, grid (using actual machine IDs) ---
            ax.set_yticks(current_plot_machine_ids) # Use actual machine IDs for ticks
            # Optionally set explicit limits if needed, e.g.:
            # ax.set_ylim(min(current_plot_machine_ids) - 1, max(current_plot_machine_ids) + 1)
            ax.set_xlabel("Time")
            ax.set_ylabel("Machine ID")
            # Title reflects the model and the machine ID range for clarity
            ax.set_title(f"Model {model_name} - Machine Events (IDs: {current_plot_machine_ids[0]} to {current_plot_machine_ids[-1]})")
            ax.grid(True, axis='x', linestyle='--', alpha=0.6)

            plt.tight_layout(rect=[0, 0, 0.80, 1])

            # --- Filename Generation --- 
            # Use model name and the *within-model index* range for the filename suffix
            plot_filename = os.path.join(output_dir, f"events_{model_name}_{start_model_index + 1}-{end_model_index}.png")
            plt.savefig(plot_filename)
            plt.close(fig)
            print(f"Saved plot: {plot_filename}")


def visualize_single_machine_telemetry(df_telemetry, combined_events, machine_id, num_days=None, output_dir="plots"):
    """
    Creates a time series plot for a single machine showing telemetry data 
    (volt, rotate, pressure, vibration) and overlays events (errors, failures,
    maintenance) as vertical lines with text annotations.
    Can optionally filter to the first num_days of 2014.

    Parameters:
    -----------
    df_telemetry : pandas.DataFrame
        DataFrame containing telemetry data with 'datetime', 'machineID', 
        'volt', 'rotate', 'pressure', 'vibration'.
    combined_events : pandas.DataFrame
        DataFrame containing combined event data with 'datetime', 'machineID', 
        'eventType', 'eventcategory'.
    machine_id : int
        The ID of the machine to plot.
    num_days : int, optional
        If provided, plots only the first num_days starting from 2014-01-01.
        If None (default), plots all data with events filtered from 2015-01-01.
    output_dir : str, optional
        Directory to save the plot, by default "plots"
    """
    # --- Input Validation ---
    telemetry_required_cols = ['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration']
    if not all(col in df_telemetry.columns for col in telemetry_required_cols):
        print(f"Error: Telemetry DataFrame missing one or more required columns: {telemetry_required_cols}")
        return

    event_required_cols = ['datetime', 'machineID', 'eventType', 'eventcategory']
    if not all(col in combined_events.columns for col in event_required_cols):
        print(f"Error: Combined Events DataFrame missing one or more required columns: {event_required_cols}")
        # Proceeding without events if telemetry is valid
        plot_events = False
    else:
        plot_events = True

    # --- Filter Data by Machine ID ---
    machine_telemetry = df_telemetry[df_telemetry['machineID'] == machine_id].copy()
    if machine_telemetry.empty:
        print(f"Error: No telemetry data found for machineID {machine_id}.")
        return

    if plot_events:
        machine_events = combined_events[combined_events['machineID'] == machine_id].copy()
        # Ensure datetime is in the correct format before filtering
        machine_events['datetime'] = pd.to_datetime(machine_events['datetime'])
    else:
        machine_events = pd.DataFrame()

    # --- Apply Time Period Filter --- 
    plot_title = f"Telemetry Data and Events for Machine ID {machine_id}"
    filename_suffix = ""

    # Ensure telemetry datetime is correct type before filtering
    machine_telemetry['datetime'] = pd.to_datetime(machine_telemetry['datetime'])

    if num_days is not None:
        try:
            num_days = int(num_days)
            start_date = pd.Timestamp('2015-01-01')
            end_date = start_date + pd.Timedelta(days=num_days)
            
            print(f"Filtering data for machine {machine_id} to {start_date.date()} - {end_date.date()}")
            
            # Filter telemetry
            machine_telemetry = machine_telemetry[
                (machine_telemetry['datetime'] >= start_date) & 
                (machine_telemetry['datetime'] < end_date)
            ]
            
            # Filter events (if they exist)
            if plot_events:
                 machine_events = machine_events[
                    (machine_events['datetime'] >= start_date) & 
                    (machine_events['datetime'] < end_date)
                ]
            
            plot_title = f"Telemetry Data and Events for Machine ID {machine_id} (First {num_days} Days of 2014)"
            filename_suffix = f"_2014_first_{num_days}_days"

        except ValueError:
             print(f"Warning: Invalid value provided for num_days ('{num_days}'). Plotting full history instead.")
             # Apply default event filtering if num_days was invalid
             if plot_events:
                start_date_events = pd.Timestamp('2015-01-01')
                machine_events = machine_events[machine_events['datetime'] >= start_date_events]
                print(f"Filtered events for machine {machine_id} to on/after {start_date_events.date()}")
    else:
        # Default behavior: Filter events from 2015 onwards
        if plot_events:
            start_date_events = pd.Timestamp('2015-01-01')
            machine_events = machine_events[machine_events['datetime'] >= start_date_events]
            print(f"Filtered events for machine {machine_id} to on/after {start_date_events.date()}")
            
    # --- Filter out maintenance events (applied AFTER date range selection) --- 
    if plot_events:
        machine_events = machine_events[machine_events['eventType'] != 'maintenance']
        print(f"Filtered out maintenance events for machine {machine_id}.")
        machine_events['eventcategory'] = machine_events['eventcategory'].astype(str)

    # --- Prepare Telemetry for Plotting (after potential date filtering) ---
    if machine_telemetry.empty:
        print(f"Error: No telemetry data found for machineID {machine_id} in the specified period.")
        return
        
    machine_telemetry.set_index('datetime', inplace=True)
    telemetry_cols = ['volt', 'rotate', 'pressure', 'vibration']
    
    print(f"\nPlotting telemetry and events for machine {machine_id}...")
    
    # --- Create Plot ---
    fig, ax = plt.subplots(figsize=(17, 8))
    
    # Plot Telemetry Data
    machine_telemetry[telemetry_cols].plot(ax=ax, legend=True)
    
    # --- Plot Events ---
    if plot_events and not machine_events.empty:
        y_min, y_max = ax.get_ylim() # Get y-axis limits for text placement
        text_y_pos = y_max * 0.95 # Position text near the top
        
        for _, event in machine_events.iterrows():
            event_time = event['datetime']
            event_label = f"{event['eventType']}_{event['eventcategory']}"
            
            # Draw vertical line
            ax.axvline(x=event_time, color='black', linestyle='--', linewidth=1)
            
            # Add text annotation
            ax.text(x=event_time, y=text_y_pos, s=event_label, 
                    rotation=90, verticalalignment='top', 
                    fontsize=8, color='black', 
                    bbox=dict(facecolor='white', alpha=0.5, pad=0.2, boxstyle='round,pad=0.3'))

    # --- Finalize Plot ---
    ax.set_xlabel("Time")
    ax.set_ylabel("Sensor Value / Event Marker")
    ax.set_title(plot_title) # Use dynamic title
    ax.legend(title='Sensor', loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Update filename based on filtering
    plot_filename = os.path.join(output_dir, f"telemetry_events_machine_{machine_id}{filename_suffix}.png") 
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig) 
    print(f"Saved telemetry and events plot: {plot_filename}")

def visualize_event_category_percentage(df, machine_id, output_dir="plots"):
    """
    Creates a stacked horizontal bar chart showing the percentage distribution 
    of non-maintenance event categories (errors vs failures) for a specific machine.

    Parameters:
    -----------
    df : pandas.DataFrame
        Combined DataFrame with at least 'machineID', 'eventcategory', 'eventType'.
    machine_id : int
        The ID of the machine to plot.
    output_dir : str, optional
        Directory to save the plot, by default "plots"
    """
    required_cols = ["machineID", "eventcategory", "eventType"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame missing one or more required columns: {required_cols}.")
        return

    # Filter for the specific machine
    machine_data = df[df["machineID"] == machine_id].copy()

    if machine_data.empty:
        print(f"Error: No event data found for machineID {machine_id}.")
        return

    # Filter out maintenance events
    non_maint_data = machine_data[machine_data["eventType"] != "maintenance"].copy()

    if non_maint_data.empty:
        print(f"Error: No error or failure data found for machineID {machine_id}.")
        return

    # Calculate counts and percentages for event categories, grouped by type
    total_non_maint_events = len(non_maint_data)
    
    category_type_counts = non_maint_data.groupby(["eventcategory", "eventType"]).size()
    category_type_percentages = (category_type_counts / total_non_maint_events) * 100

    # Unstack to get categories as index and types (error/failure) as columns
    percentage_df = category_type_percentages.unstack(fill_value=0)
    
    # Ensure both 'error' and 'failure' columns exist, even if one has no events
    if 'error' not in percentage_df.columns:
        percentage_df['error'] = 0
    if 'failure' not in percentage_df.columns:
        percentage_df['failure'] = 0
        
    # Reorder columns for consistent legend (optional)
    percentage_df = percentage_df[['error', 'failure']]

    # Calculate total percentage per category for sorting
    percentage_df["total_perc"] = percentage_df["error"] + percentage_df["failure"]
    percentage_df = percentage_df.sort_values("total_perc", ascending=True)

    print(f"\nEvent Category Percentage (Errors/Failures) for Machine {machine_id}:")
    print(percentage_df)

    # Plotting the stacked horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, max(6, len(percentage_df) * 0.5))) 
    
    # Define colors
    colors = {"error": "#d62728", "failure": "#1f77b4"} # Red for error, Blue for failure
    
    percentage_df[["error", "failure"]].plot(kind="barh", stacked=True, color=[colors["error"], colors["failure"]], ax=ax)

    # Set labels and title
    ax.set_xlabel(f"Percentage of Total Non-Maintenance Events ({total_non_maint_events} events)")
    ax.set_ylabel("Event Category")
    ax.set_title(f"Error/Failure Category Distribution (%) for Machine ID {machine_id}")
    ax.legend(title="Event Type")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Add total percentage labels to the bars (optional)
    for i, total_perc in enumerate(percentage_df["total_perc"]):
        if total_perc > 0: # Only label bars with values
             ax.text(total_perc + 0.5, i, f'{total_perc:.1f}%', va='center', fontsize=9)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"event_error_failure_percentage_machine_{machine_id}.png")
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig) # Close the figure
    print(f"Saved error/failure category percentage plot: {plot_filename}")

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
    
    # Add mean/median lines (optional)
    mean_duration = durations.mean()
    median_duration = durations.median()
    ax.axvline(mean_duration, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_duration:.1f} {unit}')
    ax.axvline(median_duration, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_duration:.1f} {unit}')
    ax.legend()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"maintenance_duration_histogram_{unit}.png")
    
    plt.tight_layout()
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved maintenance duration histogram: {plot_filename}")

def visualize_detrended_periodogram(df_telemetry, machine_id, output_dir="plots"):
    """
    Detrends telemetry signals for a specific machine and plots their periodograms.

    Parameters:
    -----------
    df_telemetry : pandas.DataFrame
        DataFrame containing telemetry data with 'datetime', 'machineID', 
        'volt', 'rotate', 'pressure', 'vibration'.
    machine_id : int
        The ID of the machine to analyze.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    """
    required_cols = ['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration']
    if not all(col in df_telemetry.columns for col in required_cols):
        print(f"Error: Telemetry DataFrame missing one or more required columns: {required_cols}")
        return

    # Filter for the specific machine and sort by time
    machine_data = df_telemetry[df_telemetry['machineID'] == machine_id].copy()
    machine_data['datetime'] = pd.to_datetime(machine_data['datetime'])
    machine_data.sort_values(by='datetime', inplace=True)

    if machine_data.empty:
        print(f"Error: No telemetry data found for machineID {machine_id}.")
        return
        
    # Check for NaNs - detrend/periodogram might fail or give misleading results
    if machine_data[['volt', 'rotate', 'pressure', 'vibration']].isnull().values.any():
        print(f"Warning: NaN values found in telemetry data for machine {machine_id}. Consider imputation before analysis.")
        # Option: Fill NaNs, e.g., machine_data.fillna(method='ffill', inplace=True) or drop them
        # For now, we proceed but results might be affected.
        
    print(f"\nGenerating periodograms for machine {machine_id}...")

    telemetry_cols = ['volt', 'rotate', 'pressure', 'vibration']
    # Assume sampling frequency is 1 (per record/hour). Adjust if known otherwise.
    fs = 1 

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten() # Flatten the 2x2 array for easy iteration

    for i, col in enumerate(telemetry_cols):
        ax = axes[i]
        signal_data = machine_data[col].values
        
        # Detrend the signal
        try:
            detrended_signal = signal.detrend(signal_data)
            # Calculate periodogram
            f, Pxx = signal.periodogram(detrended_signal, fs=fs)
            
            # Plot periodogram (Power Spectral Density)
            ax.semilogy(f, Pxx) # Use log scale for power
            ax.set_title(f"Detrended {col.capitalize()}")
            ax.set_xlabel("Frequency (cycles/sample)")
            ax.set_ylabel("PSD (Power/Frequency)")
            ax.grid(True, linestyle='--', alpha=0.6)
            # Limit x-axis to Nyquist frequency if needed (f <= fs/2)
            # ax.set_xlim([0, fs / 2]) 
            
        except ValueError as e:
             print(f"Could not process signal {col} for machine {machine_id}. Error: {e}")
             ax.set_title(f"Detrended {col.capitalize()} (Error)")
             ax.text(0.5, 0.5, "Error during processing", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


    fig.suptitle(f"Periodograms of Detrended Telemetry Signals for Machine ID {machine_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f"periodogram_machine_{machine_id}.png")
    
    plt.savefig(plot_filename)
    plt.close(fig)
    print(f"Saved periodogram plot: {plot_filename}")

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
        print("\nCombined and Enriched Events DataFrame head:")
        print(combined_events.head())
        print("\nColumns:", combined_events.columns)

        # --- Visualize Data --- 
        # Event Timeseries visualization
        # visualize_timeseries(combined_events) 
   

        # Telemetry visualization for Machine 1 with Events
        visualize_single_machine_telemetry(PDM_Telemetry, combined_events, machine_id=1, num_days=14) # Default: full history
        # Example: visualize_single_machine_telemetry(PDM_Telemetry, combined_events, machine_id=1, num_days=90) # First 90 days of 2014
        
        # Event Category Percentage visualization for Machine 1
        visualize_event_category_percentage(combined_events, machine_id=1)
        
        # Maintenance Duration Histogram
        visualize_maintenance_durations(Maintenance, unit='days') # Calculate in days
        
        # Visualize Periodogram for Machine 1
        visualize_detrended_periodogram(PDM_Telemetry, machine_id=1)

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