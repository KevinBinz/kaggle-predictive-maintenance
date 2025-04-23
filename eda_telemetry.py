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

def plot_single_machine_telemetry(df_telemetry, combined_events, machine_id, num_days=None, output_dir="plots"):
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

def detrend_telemetry(df_telemetry):
    """
    Detrends the telemetry signal columns ('volt', 'rotate', 'pressure', 'vibration')
    in the input DataFrame.

    Parameters:
    -----------
    df_telemetry : pandas.DataFrame
        DataFrame containing telemetry data with required columns.

    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with detrended telemetry columns. Original columns
        are kept if detrending fails for a specific column.
    """
    required_cols = ['volt', 'rotate', 'pressure', 'vibration']
    if not all(col in df_telemetry.columns for col in required_cols):
        print(f"Error: Detrending requires columns: {required_cols}. Skipping.")
        return df_telemetry.copy() # Return original if columns missing

    df_detrended = df_telemetry.copy()
    
    print("\nDetrending telemetry signals...")

    for col in required_cols:
        signal_data = df_detrended[col].values
        
        # Check for NaNs or constant signals before detrending
        if np.isnan(signal_data).all():
             print(f"Skipping detrending for {col}: all values are NaN.")
             continue
        if len(np.unique(signal_data[~np.isnan(signal_data)])) <= 1:
             print(f"Skipping detrending for {col}: signal is constant (or NaN).")
             continue
             
        try:
            # Handle potential NaNs by detrending only non-NaN parts if necessary,
            # though scipy.signal.detrend might handle internal NaNs depending on version.
            # A simple approach: detrend and assign back, assuming detrend handles it.
            # More robust: interpolate NaNs before detrending.
            detrended = signal.detrend(signal_data)
            df_detrended[col] = detrended - detrended.mean()
            print(f"Detrended {col}. Mean={df_detrended[col].mean()} std={df_detrended[col].std()}")
        except ValueError as e:
            print(f"Could not detrend signal {col}. Keeping original. Error: {e}")
            # Keep original column if detrending fails
        

    return df_detrended

def plot_detrended_periodogram(df_telemetry, machine_id, output_dir="plots"):
    """
    Detrends telemetry signals for a specific machine and plots their periodograms
    (standard and Welch method).

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

    # --- Standard Periodogram Plot ---
    fig_std, axes_std = plt.subplots(2, 2, figsize=(12, 10))
    axes_std = axes_std.flatten() # Flatten the 2x2 array for easy iteration

    # --- Welch Periodogram Plot ---
    fig_welch, axes_welch = plt.subplots(2, 2, figsize=(12, 10))
    axes_welch = axes_welch.flatten()

    all_signals_processed_std = True
    all_signals_processed_welch = True

    for i, col in enumerate(telemetry_cols):
        ax_std = axes_std[i]
        ax_welch = axes_welch[i]
        signal_data = machine_data[col].values
        
        # Detrend the signal
        try:
            detrended_signal = signal.detrend(signal_data)

            # --- Standard Periodogram ---
            try:
                f_std, Pxx_std = signal.periodogram(detrended_signal, fs=fs, detrend='constant')
                # Skip the zero-frequency component (f[0])
                f_std_plot = f_std[1:]
                Pxx_std_plot = Pxx_std[1:]
                
                # Plot periodogram (Power Spectral Density)
                ax_std.semilogy(f_std_plot, Pxx_std_plot) # Use log scale for power
                ax_std.set_title(f"Standard Periodogram: {col.capitalize()}")
                ax_std.set_xlabel("Frequency (cycles/sample)")
                ax_std.set_ylabel("PSD (Power/Frequency)")
                ax_std.grid(True, linestyle='--', alpha=0.6)
                # ax_std.set_xlim([0, fs / 2]) # Optionally limit x-axis
            except ValueError as e_std:
                print(f"Could not calculate standard periodogram for {col}, machine {machine_id}. Error: {e_std}")
                ax_std.set_title(f"Standard Periodogram: {col.capitalize()} (Error)")
                ax_std.text(0.5, 0.5, "Error during processing", horizontalalignment='center', verticalalignment='center', transform=ax_std.transAxes)
                all_signals_processed_std = False
            
            # --- Welch Periodogram ---
            try:
                f_welch, Pxx_welch = signal.welch(
                    detrended_signal,
                    fs=24,              # 24 samples per day
                    detrend='constant',
                    window='hann',
                    nperseg=24*7,       # e.g. 1‑week segments for smoother low‑freq resolution
                    noverlap=0
                )
                # Skip the zero-frequency component (f[0])
                f_welch_plot = f_welch[1:]
                Pxx_welch_plot = Pxx_welch[1:]

                # Plot Welch periodogram (Power Spectral Density)
                ax_welch.semilogy(f_welch, Pxx_welch) # Use log scale for power
                ax_welch.set_title(f"Welch Periodogram: {col.capitalize()}")
                ax_welch.set_xlabel("Frequency (cycles/sample)")
                ax_welch.set_ylabel("PSD (Power/Frequency)")
                ax_welch.grid(True, linestyle='--', alpha=0.6)
                # ax_welch.set_xlim([0, fs / 2]) # Optionally limit x-axis
            except ValueError as e_welch:
                print(f"Could not calculate Welch periodogram for {col}, machine {machine_id}. Error: {e_welch}")
                ax_welch.set_title(f"Welch Periodogram: {col.capitalize()} (Error)")
                ax_welch.text(0.5, 0.5, "Error during processing", horizontalalignment='center', verticalalignment='center', transform=ax_welch.transAxes)
                all_signals_processed_welch = False

        except ValueError as e_detrend:
             print(f"Could not detrend signal {col} for machine {machine_id}. Skipping periodograms. Error: {e_detrend}")
             # Mark both plots as error for this signal
             ax_std.set_title(f"Standard Periodogram: {col.capitalize()} (Detrend Error)")
             ax_std.text(0.5, 0.5, "Error during detrending", ha='center', va='center', transform=ax_std.transAxes)
             ax_welch.set_title(f"Welch Periodogram: {col.capitalize()} (Detrend Error)")
             ax_welch.text(0.5, 0.5, "Error during detrending", ha='center', va='center', transform=ax_welch.transAxes)
             all_signals_processed_std = False
             all_signals_processed_welch = False

    # --- Finalize and Save Standard Periodogram Plot ---
    if all_signals_processed_std:
        fig_std.suptitle(f"Standard Periodograms of Detrended Telemetry Signals for Machine ID {machine_id}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        plot_filename_std = os.path.join(output_dir, f"telemetry_periodogram_machine_{machine_id}.png")
        plt.figure(fig_std.number) # Ensure we save the correct figure
        plt.savefig(plot_filename_std)
        print(f"Saved standard periodogram plot: {plot_filename_std}")
    else:
        print(f"Standard periodogram plot not saved due to processing errors for machine {machine_id}.")
    plt.close(fig_std)

    # --- Finalize and Save Welch Periodogram Plot ---
    if all_signals_processed_welch:
        fig_welch.suptitle(f"Welch Periodograms of Detrended Telemetry Signals for Machine ID {machine_id}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        plot_filename_welch = os.path.join(output_dir, f"telemetry_welch_periodogram_machine_{machine_id}.png")
        plt.figure(fig_welch.number) # Ensure we save the correct figure
        plt.savefig(plot_filename_welch)
        print(f"Saved Welch periodogram plot: {plot_filename_welch}")
    else:
         print(f"Welch periodogram plot not saved due to processing errors for machine {machine_id}.")
    plt.close(fig_welch)


def check_daily_rhythm(df, output_dir="plots"):
    # import matplotlib.pyplot as plt # Moved to top
    # import pandas as pd # Moved to top
    # from statsmodels.graphics.tsaplots import plot_acf # Moved to top
    # from statsmodels.tsa.seasonal import seasonal_decompose # Moved to top

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Ensure datetime is the index, but work on a copy to avoid modifying original df outside the function
    df_indexed = df.set_index('datetime')

    telemetry_cols = ['volt', 'rotate', 'pressure', 'vibration']
    print(f"\nChecking daily rhythm for columns: {telemetry_cols}")

    for col in telemetry_cols:
        # Check if column exists
        if col not in df_indexed.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping rhythm check for this column.")
            continue

        print(f"Processing column: {col}...")
        ts = df_indexed[col].dropna() # Drop NaN values which seasonal_decompose might handle poorly
        acf24 = ts.autocorr(lag=24)
        print(f"ACF at 24 h: {acf24:.3f}")


        if ts.empty:
            print(f"Warning: Column '{col}' contains only NaN values after dropping. Skipping.")
            continue
        if len(ts) < 2 * 24: # Need at least two full periods for decomposition
            print(f"Warning: Not enough data ({len(ts)} points) in column '{col}' for seasonal decomposition with period 24. Skipping decomposition.")
        else:
            try:
                result = seasonal_decompose(ts, model='additive', period=24)

                # 1) Trend: weekly down‑sample (mean per week)
                trend_weekly = result.trend.resample('7D').mean()
                seasonal_24h = result.seasonal.iloc[:24].reset_index(drop=True)
                resid = result.resid.dropna()
                fig_decomp, axes = plt.subplots(3,1, figsize=(10, 8), constrained_layout=True)

                # Trend panel
                axes[0].plot(trend_weekly.index, trend_weekly.values, marker='o', linestyle='-')
                axes[0].set_title('Trend (weekly average)')
                axes[0].set_ylabel(ts.name)

                # Seasonal panel
                axes[1].plot(np.arange(24), seasonal_24h, marker='o', linestyle='-')
                axes[1].set_xticks(range(0,24,3))
                axes[1].set_xlabel('Hour of day')
                axes[1].set_ylabel('Seasonal')
                axes[1].set_title('24 h Seasonal Component')

                # Residuals panel
                axes[2].hist(resid, bins=50, edgecolor='k')
                axes[2].set_title('Residuals Distribution')
                axes[2].set_xlabel('Residual')
                axes[2].set_ylabel('Count')
                plot_filename_decomp = os.path.join(output_dir, f"telemetry_seasonal_{col}.png")
                plt.savefig(plot_filename_decomp)
                print(f"Saved seasonal decomposition plot: {plot_filename_decomp}")
                plt.close(fig_decomp) # Close the correct figure
            except ValueError as e:
                print(f"Error during seasonal decomposition for column '{col}': {e}")
            except Exception as e:
                 print(f"An unexpected error occurred during seasonal decomposition for column '{col}': {e}")

        # Plot ACF
        if len(ts) > 48: # Ensure enough data for ACF lags
            try:
                fig_acf = plt.figure(figsize=(8,4)) # Capture the figure handle
                plot_acf(ts, lags=48, zero=False, ax=fig_acf.gca()) # Pass axis to plot_acf
                plt.axvline(24, color='red', linestyle='--', label='24h lag')
                plt.title(f"Autocorrelation for {col} (first 48 lags)")
                plt.legend()
                plot_filename_acf = os.path.join(output_dir, f"telemetry_autocorrelation_{col}.png")
                plt.savefig(plot_filename_acf)
                print(f"Saved autocorrelation plot: {plot_filename_acf}")
                plt.close(fig_acf) # Close the correct figure
            except Exception as e:
                print(f"An unexpected error occurred during ACF plotting for column '{col}': {e}")
                # Ensure figure is closed even if error occurs before saving
                if 'fig_acf' in locals() and plt.fignum_exists(fig_acf.number):
                    plt.close(fig_acf)
        else:
            print(f"Warning: Not enough data ({len(ts)} points) in column '{col}' for ACF plot with 48 lags. Skipping ACF plot.")


def plot_wavelet_transform(df_telemetry, machine_id, output_dir="plots", wavelet_name='morl', max_scale=64):
    """
    Performs Continuous Wavelet Transform (CWT) on telemetry signals for a specific
    machine and plots the scalograms (magnitude of CWT coefficients).

    Requires the 'PyWavelets' library (pip install PyWavelets).
    Requires 'numpy' and 'matplotlib'.

    Parameters:
    -----------
    df_telemetry : pandas.DataFrame
        DataFrame containing telemetry data with 'datetime', 'machineID',
        'volt', 'rotate', 'pressure', 'vibration'.
    machine_id : int
        The ID of the machine to analyze.
    output_dir : str, optional
        Directory to save the plot, by default "plots".
    wavelet_name : str, optional
        Name of the wavelet to use (e.g., 'morl', 'cmor', 'gaus1'), by default 'morl'.
    max_scale : int, optional
        Maximum scale to use for the CWT analysis, by default 64.
        Higher scales correspond to lower frequencies.
    """
    # Local imports to make the function self-contained if needed, though top-level is preferred.

    required_cols = ['datetime', 'machineID', 'volt', 'rotate', 'pressure', 'vibration']
    if not all(col in df_telemetry.columns for col in required_cols):
        print(f"Error: Telemetry DataFrame missing one or more required columns: {required_cols}")
        return

    # Filter for the specific machine and sort by time
    machine_data = df_telemetry[df_telemetry['machineID'] == machine_id].copy()
    # Ensure datetime column is in right format if not already
    if not pd.api.types.is_datetime64_any_dtype(machine_data['datetime']):
        machine_data['datetime'] = pd.to_datetime(machine_data['datetime'])
    machine_data.sort_values(by='datetime', inplace=True)

    if machine_data.empty:
        print(f"Error: No telemetry data found for machineID {machine_id}.")
        return

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    telemetry_cols = ['volt', 'rotate', 'pressure', 'vibration']
    # Define scales for CWT. A simple range from 1 to max_scale.
    scales = np.arange(1, max_scale + 1)
    # Assuming data is hourly, sampling period dt = 1 hour.
    dt = 1

    print(f"\nGenerating CWT scalograms for machine {machine_id} using '{wavelet_name}' wavelet...")

    fig, axes = plt.subplots(len(telemetry_cols), 1, figsize=(15, 4 * len(telemetry_cols)), sharex=True)
    if len(telemetry_cols) == 1:
        axes = [axes] # Ensure axes is iterable even for one plot

    time_index = machine_data['datetime'] # For x-axis
    time_num = np.arange(len(time_index)) # Numerical time for extent

    all_signals_processed = True

    for i, col in enumerate(telemetry_cols):
        ax = axes[i]
        signal_data = machine_data[col].values

        # Check for NaNs - CWT typically requires continuous data
        if np.isnan(signal_data).any():
            print(f"Warning: NaN values found in {col} for machine {machine_id}. CWT might produce unexpected results or fail. Consider imputation.")
            # Optional: Interpolate NaNs
            # signal_data = pd.Series(signal_data).interpolate().values
            # Check again if interpolation left NaNs (e.g., at edges)
            if np.isnan(signal_data).any():
                 print(f"Error: NaN values still present in {col} after potential handling. Skipping CWT.")
                 ax.set_title(f"{col.capitalize()} - NaN Values Present")
                 ax.text(0.5, 0.5, "NaN values prevent CWT", ha='center', va='center', transform=ax.transAxes)
                 all_signals_processed = False
                 continue

        if len(signal_data) < max(scales): # Check if signal is long enough for chosen scales
            print(f"Warning: Signal length ({len(signal_data)}) for {col} is shorter than max scale ({max(scales)}). Reducing max scale.")
            current_max_scale = len(signal_data) - 1
            if current_max_scale < 1:
                 print(f"Error: Signal {col} too short for CWT. Skipping.")
                 ax.set_title(f"{col.capitalize()} - Signal Too Short for CWT")
                 ax.text(0.5, 0.5, "Signal too short", ha='center', va='center', transform=ax.transAxes)
                 all_signals_processed = False
                 continue
            current_scales = np.arange(1, current_max_scale + 1)
        else:
            current_scales = scales

        try:
            # Perform CWT
            coefficients, frequencies = pywt.cwt(signal_data, current_scales, wavelet_name, sampling_period=dt)

            # Calculate periods corresponding to frequencies (in hours)
            # Avoid division by zero
            valid_freq_mask = frequencies > 1e-9 # Check for non-zero frequencies
            periods = np.full_like(frequencies, np.nan)
            periods[valid_freq_mask] = 1 / frequencies[valid_freq_mask]

            # Plot the scalogram (magnitude of coefficients)
            # Use periods for the y-axis, time for the x-axis
            # extent defines the x and y ranges: [left, right, bottom, top]
            # Need finite periods for extent
            valid_period_mask = np.isfinite(periods)
            if not np.any(valid_period_mask):
                 print(f"Error: No valid periods calculated for {col}. Skipping plot.")
                 ax.set_title(f"{col.capitalize()} - Period Calculation Error")
                 all_signals_processed = False
                 continue
            
            # Log scale for periods often makes sense
            # Use min/max of valid periods for y-axis limits
            # Flip y-axis so low frequency (large period) is at the bottom
            im = ax.imshow(np.abs(coefficients), cmap='viridis', aspect='auto',
                           extent=[time_num[0], time_num[-1], periods[valid_period_mask].max(), periods[valid_period_mask].min()])

            ax.set_title(f"Wavelet Scalogram: {col.capitalize()}")
            ax.set_ylabel("Period (hours)")

            # Configure y-axis ticks and labels
            y_min_lim, y_max_lim = periods[valid_period_mask].min(), periods[valid_period_mask].max()
            ax.set_ylim(y_max_lim, y_min_lim) # Flipped y-axis

            if periods[valid_period_mask].max() / periods[valid_period_mask].min() > 10:
                ax.set_yscale('log')
                # Use LogLocator for automatic log ticks
                from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter, NullLocator
                # Set major ticks for powers of 2 or 10, or automatically
                ax.yaxis.set_major_locator(LogLocator(base=2, numticks=10))
                ax.yaxis.set_major_formatter(ScalarFormatter()) # Use standard number format
                # Remove minor ticks by setting locator to NullLocator
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                 # Linear scale: Define a reasonable number of ticks
                 from matplotlib.ticker import MaxNLocator
                 ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=False))
                 ax.yaxis.set_major_formatter(ScalarFormatter())

            # Set x-ticks to be dates
            # Increase the number of ticks for better visibility
            num_ticks = 20 
            tick_indices = np.linspace(0, len(time_index) - 1, num=num_ticks, dtype=int)
            ax.set_xticks(time_num[tick_indices])
            ax.set_xticklabels(time_index.iloc[tick_indices].dt.strftime('%Y-%m-%d'), rotation=45, ha='right')

            # Add horizontal line at 24-hour period if within y-limits
            y_min, y_max = ax.get_ylim()
            if y_min <= 24 <= y_max:
                ax.axhline(24, color='red', linestyle='--', linewidth=1, label='24h Period')
                # Ensure legend is shown if label is added
                if i == 0: # Add legend only once or handle repetition
                     ax.legend(loc='upper right')
            elif i == 0: # If 24h is not plotted, maybe remove placeholder legend if any was created
                 # Check if legend exists before trying to remove (might not be necessary)
                 if ax.get_legend():
                     ax.get_legend().remove()

            # Add colorbar
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label('Magnitude')

        except ValueError as e:
            print(f"Could not perform CWT for signal {col}, machine {machine_id}. Error: {e}")
            ax.set_title(f"{col.capitalize()} - CWT Error")
            ax.text(0.5, 0.5, f"Error during CWT: {e}", ha='center', va='center', transform=ax.transAxes, wrap=True)
            all_signals_processed = False
        except Exception as e:
            print(f"An unexpected error occurred during CWT for {col}, machine {machine_id}: {e}")
            ax.set_title(f"{col.capitalize()} - Unexpected CWT Error")
            ax.text(0.5, 0.5, f"Unexpected Error: {e}", ha='center', va='center', transform=ax.transAxes, wrap=True)
            all_signals_processed = False

    axes[-1].set_xlabel("Time") # Set xlabel only on the last subplot
    fig.suptitle(f"Continuous Wavelet Transform (Scalograms) for Machine ID {machine_id}", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout

    # Save the plot
    plot_filename_base = f"telemetry_wavelet_scalogram_machine_{machine_id}"
    if not all_signals_processed:
        print(f"CWT scalogram plot for machine {machine_id} might be incomplete due to processing errors.")
        plot_filename = os.path.join(output_dir, f"{plot_filename_base}_partial.png")
    else:
        plot_filename = os.path.join(output_dir, f"{plot_filename_base}.png")

    plt.savefig(plot_filename)
    print(f"Saved CWT scalogram plot: {plot_filename}")
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
        
        plot_single_machine_telemetry(PDM_Telemetry, combined_events, machine_id=1, num_days=None) # Default: full history
        detrended_telemetry = detrend_telemetry(PDM_Telemetry)
        plot_detrended_periodogram(detrended_telemetry, machine_id=1)
        #check_daily_rhythm(PDM_Telemetry)
        plot_wavelet_transform(PDM_Telemetry, machine_id=1)

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Make sure the CSV files are in the '{data_path}' directory.")
    except KeyError as e:
        print(f"Error processing/visualizing data: Missing expected column {e}. Check CSV file structures and processing steps.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
