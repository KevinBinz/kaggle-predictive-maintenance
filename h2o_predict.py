import pandas as pd
import numpy as np
import os
import h2o
from h2o.automl import H2OAutoML
from data_utilities import split_data_by_date, prepare_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # For generating distinct colors


def generate_preds():
    non_maintenance_df = prepare_data(only_maintenance=False, 
                                       lag_list=[1, 168],
                                       agg_list=['median', 'max', 'min', 'var'])
    non_maintenance_df.drop(columns=['daysSinceLastMaintenanceEvent'], inplace=True)
    # print(non_maintenance_df['daysSinceLastMaintenanceEvent'].head())
    # non_maintenance_df['daysSinceLastMaintenanceEvent'].astype(int)
    # print(non_maintenance_df['daysSinceLastMaintenanceEvent'].head())

    target = "isFailureEvent"
    n = 10
    df_chunks = np.array_split(non_maintenance_df, n)

    h2o.init(max_mem_size="2G") # Increased memory allocation
    model_path = "data/DeepLearning_1_AutoML_1_20250426_145520"
    loaded_model = h2o.load_model(model_path)
    preds = []
    for i, chunk in enumerate(df_chunks):
        chunk_hf = h2o.H2OFrame(chunk)
        preds.append(loaded_model.predict(chunk_hf).as_data_frame(use_multi_thread=True))

    combined_data = pd.concat(df_chunks)
    combined_preds = pd.concat(preds)

    combined_data = combined_data.reset_index(drop=True)
    combined_preds = combined_preds.reset_index(drop=True)

    print(f"Data.shape={combined_data.shape} Preds.shape={combined_preds.shape}")
    combined = pd.concat([combined_data, combined_preds], axis=1)
    print(f"Concat.shape={combined.shape}")
    print(combined.head())
    combined.to_csv('combined.csv', index=False)

    h2o.cluster().shutdown() # Shutdown H2O cluster when done
    print("\nH2O cluster shutdown.")

def contextualize_preds():
    # Load predictions made on non_maintenance data
    non_maintenance_preds = pd.read_csv('combined.csv')
    print(f"Loaded predictions shape: {non_maintenance_preds.shape}")
    print(f"Prediction columns: {non_maintenance_preds.columns.tolist()}")

    # Prepare the ground truth data (only_maintenance)
    only_maintenance_df = prepare_data(only_maintenance=True, 
                                       lag_list=[1, 168],
                                       agg_list=['median', 'max', 'min', 'var'])
    # Ensure correct columns are dropped based on prepare_data logic
    # Note: daysSinceLastMaintenanceEvent might be dropped inside prepare_data if it matches exclude criteria
    if 'daysSinceLastMaintenanceEvent' in only_maintenance_df.columns:
        only_maintenance_df = only_maintenance_df.drop(columns=['daysSinceLastMaintenanceEvent'])
    print(f"Loaded ground truth shape: {only_maintenance_df.shape}")
    print(f"Ground truth columns: {only_maintenance_df.columns.tolist()}")

    # --- Prepare Prediction Data --- 
    keep_features = ['machineID', 'datetime', 'volt', 'rotate', 'pressure', 'vibration']
    preds_selected = non_maintenance_preds[keep_features + ['predict']].copy()
    preds_selected.rename(columns={'predict': 'LabelValue'}, inplace=True)
    preds_selected['LabelType'] = 'LabelPrediction'
    preds_selected['LabelValue'] = preds_selected['LabelValue'].astype(bool)

    # --- Prepare Ground Truth Data --- 
    truth_selected = only_maintenance_df[keep_features + ['isFailureEvent']].copy()
    truth_selected.rename(columns={'isFailureEvent': 'LabelValue'}, inplace=True)
    truth_selected['LabelType'] = 'GroundTruth'
    truth_selected['LabelValue'] = truth_selected['LabelValue'].astype(bool) 

    # --- Union the two dataframes --- 
    combined_labels = pd.concat([truth_selected, preds_selected], ignore_index=True)

    print(f"\nCombined Labels DataFrame shape: {combined_labels.shape}")
    print("Combined Labels DataFrame head:")
    print(combined_labels.head())
    print(f"\nValue counts for LabelType:\n{combined_labels['LabelType'].value_counts()}")
    print(f"\nValue counts for LabelValue:\n{combined_labels['LabelValue'].value_counts()}")
    
    # Save the combined DataFrame
    output_filename = 'combined_labels.csv'
    combined_labels.to_csv(output_filename, index=False)
    print(f"\nCombined labels saved to {output_filename}")
    return output_filename # Return the filename for the next function

def visualize_preds(combinedlabel_fn, machine_id):
    """Loads combined labels and plots telemetry data colored by label type/value 
       for a specific machine using Matplotlib subplots. Creates a SEPARATE plot
       for the window around each GroundTruth_True event.

    Args:
        combinedlabel_fn (str): Filename of the CSV containing combined labels.
        machine_id (int): The ID of the machine to visualize.
    """
    print(f"\nVisualizing data for Machine ID: {machine_id} from file: {combinedlabel_fn}")
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(combinedlabel_fn)
    except FileNotFoundError:
        print(f"Error: File not found at {combinedlabel_fn}")
        return
    except Exception as e:
        print(f"Error loading file {combinedlabel_fn}: {e}")
        return

    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime']) 

    # Filter for the specific machine
    machine_df = df[df['machineID'] == machine_id].copy()

    if machine_df.empty:
        print(f"Error: No data found for Machine ID: {machine_id}")
        return

    # Create the combined color label
    machine_df['ColorLabel'] = machine_df['LabelType'] + '_' + machine_df['LabelValue'].astype(str)
    
    # Sort by datetime 
    machine_df = machine_df.sort_values('datetime')
    
    # --- Find GroundTruth_True events --- 
    ground_truth_true_times = machine_df[machine_df['ColorLabel'] == 'GroundTruth_True']['datetime']
    
    if ground_truth_true_times.empty:
        print(f"No 'GroundTruth_True' events found for Machine ID {machine_id}. No plots generated.")
        return
        
    print(f"Found {len(ground_truth_true_times)} 'GroundTruth_True' events. Generating separate plots...")

    # --- Define consistent color mapping based on all possible labels for the machine --- 
    all_unique_labels = machine_df['ColorLabel'].unique()
    if len(all_unique_labels) <= 10:
        color_map = {label: mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]] 
                     for i, label in enumerate(all_unique_labels)}
    else: 
        colors = plt.cm.get_cmap('viridis', len(all_unique_labels))
        color_map = {label: colors(i) for i, label in enumerate(all_unique_labels)}

    # --- Loop through each event and create a plot --- 
    plot_count = 0
    for event_idx, event_time in enumerate(ground_truth_true_times):
        # Define the time window for this specific event
        days_before = pd.Timedelta(days=7)
        days_after = pd.Timedelta(days=1) 
        start_time = event_time - days_before
        end_time = event_time + days_after 
        
        # Filter data for this window
        window_df = machine_df[(machine_df['datetime'] >= start_time) & (machine_df['datetime'] <= end_time)]
        
        if window_df.empty:
            print(f"Warning: No data points found within the window for event at {event_time.date()}. Skipping plot.")
            continue
        
        print(f"Plotting window for event {event_idx + 1}/{len(ground_truth_true_times)} at {event_time.date()}...")

        # --- Create Matplotlib Plot for this window --- 
        features_to_plot = ['volt', 'rotate', 'pressure', 'vibration']
        n_features = len(features_to_plot)
        
        fig, axes = plt.subplots(n_features, 1, figsize=(15, 10), sharex=True)
        event_date_str = event_time.strftime('%Y-%m-%d') # Format date for title/filename
        fig.suptitle(f"Machine ID {machine_id} - Telemetry Around Failure Event on {event_date_str}", fontsize=16)

        # Use unique labels present *within this window* for plotting
        unique_labels_in_window = window_df['ColorLabel'].unique()

        # --- Plot each feature using window data --- 
        all_handles = []
        all_labels = []
        for i, feature in enumerate(features_to_plot):
            ax = axes[i]
            # Plot each group within the window
            for label in unique_labels_in_window:
                 # Ensure the label exists in the global color map
                if label in color_map:
                    subset = window_df[window_df['ColorLabel'] == label]
                    if not subset.empty:
                        handle = ax.scatter(subset['datetime'], subset[feature], 
                                           label=label, 
                                           color=color_map[label], 
                                           s=15) # Slightly larger points 
                        if label not in all_labels:
                            all_handles.append(handle)
                            all_labels.append(label)
                else:
                     print(f"Warning: Label '{label}' not found in colormap (this shouldn't happen).") # Safety check
            
            ax.set_ylabel(feature.capitalize())
            ax.grid(True, linestyle='--', alpha=0.6)
            # Add a vertical line for the actual event time
            ax.axvline(event_time, color='red', linestyle=':', linewidth=1.5, label='GroundTruth_True Event')
            if i == 0: # Add event line label only once to avoid duplicates in legend
                if 'GroundTruth_True Event' not in all_labels:
                    # Create a dummy handle for the line to add to legend
                    line_handle = plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.5, label='GroundTruth_True Event')
                    all_handles.append(line_handle)
                    all_labels.append('GroundTruth_True Event')
            
            if i == n_features - 1: 
                ax.set_xlabel("Time")
                ax.tick_params(axis='x', rotation=30)
            else:
                 ax.tick_params(axis='x', labelbottom=False)

        # Adjust layout and add legend for *this figure*
        fig.subplots_adjust(right=0.85) 
        fig.legend(all_handles, all_labels, title="LabelType_LabelValue", 
                   loc='center left', bbox_to_anchor=(0.87, 0.5)) 

        # Save the plot with a unique name
        plot_filename = os.path.join(output_dir, f"prediction_viz_machine_{machine_id}_event_{event_date_str}.png")
        try:
            plt.savefig(plot_filename, dpi=300)
            print(f"  Plot saved to {plot_filename}")
            plot_count += 1
        except Exception as e:
            print(f"  Error saving plot: {e}")
        plt.close(fig) # Close the figure before the next iteration
    
    print(f"\nFinished generating {plot_count} plots for Machine ID {machine_id}.")

if __name__ == "__main__":
    # generate_preds() # Uncomment if you need to generate predictions first
    #combined_csv_filename = evaluate_preds()
    visualize_preds("combined_labels.csv", machine_id=1) # Visualize for Machine ID 1