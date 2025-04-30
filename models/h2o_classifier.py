import pandas as pd
import numpy as np
import os
import h2o
from h2o.automl import H2OAutoML
import matplotlib.pyplot as plt
from data_utilities import split_data_by_date, prepare_binary_classifier_trainset
import matplotlib.colors as mcolors # For generating distinct colors


def generate_preds(model, model_path):
    non_maintenance_df = prepare_binary_classifier_trainset(only_maintenance=False, 
                                       lag_list=[1, 168],
                                       agg_list=['median', 'max', 'min', 'var'])
    non_maintenance_df.drop(columns=['daysSinceLastMaintenanceEvent'], inplace=True)
    # print(non_maintenance_df['daysSinceLastMaintenanceEvent'].head())
    # non_maintenance_df['daysSinceLastMaintenanceEvent'].astype(int)
    # print(non_maintenance_df['daysSinceLastMaintenanceEvent'].head())

    target = "isFailureEvent"
    n = 10
    df_chunks = np.array_split(non_maintenance_df, n)

    if model is None:
        model = h2o.load_model(model_path)
    preds = []
    for i, chunk in enumerate(df_chunks):
        chunk_hf = h2o.H2OFrame(chunk)
        preds.append(model.predict(chunk_hf).as_data_frame(use_multi_thread=True))

    combined_data = pd.concat(df_chunks)
    combined_preds = pd.concat(preds)

    combined_data = combined_data.reset_index(drop=True)
    combined_preds = combined_preds.reset_index(drop=True)

    print(f"Data.shape={combined_data.shape} Preds.shape={combined_preds.shape}")
    combined = pd.concat([combined_data, combined_preds], axis=1)
    print(f"Concat.shape={combined.shape}")
    print(combined.head())
    combined.to_csv('h2o_predictions.csv', index=False)

    h2o.cluster().shutdown() # Shutdown H2O cluster when done
    print("\nH2O cluster shutdown.")
    return combined

def contextualize_preds(non_maintenance_preds):
    # Load predictions made on non_maintenance data
    non_maintenance_preds = pd.read_csv('h2o_predictions.csv')
    print(f"Loaded predictions shape: {non_maintenance_preds.shape}")
    print(f"Prediction columns: {non_maintenance_preds.columns.tolist()}")

    # Prepare the ground truth data (only_maintenance)
    only_maintenance_df = prepare_binary_classifier_trainset(only_maintenance=True, 
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
    return combined_labels # Return the filename for the next function

def visualize_preds(combined_labels, machine_id):
    """Loads combined labels and plots telemetry data colored by label type/value 
       for a specific machine using Matplotlib subplots. Creates a SEPARATE plot
       for the window around each GroundTruth_True event.

    Args:
        combinedlabel_fn (str): Filename of the CSV containing combined labels.
        machine_id (int): The ID of the machine to visualize.
    """
    print(f"\nVisualizing data for Machine ID: {machine_id}")
    output_dir = "predictions"

    df = combined_labels.copy()
    df['datetime'] = pd.to_datetime(df['datetime']) 
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

def visualize_all_preds(combined_labels, machine_id):
    """Loads combined labels and plots the *entire* telemetry data time series 
       colored by label type/value for a specific machine using Matplotlib subplots.

    Args:
        combined_labels (pd.DataFrame): DataFrame containing combined labels and telemetry.
        machine_id (int): The ID of the machine to visualize.
    """
    print(f"\nVisualizing ALL data for Machine ID: {machine_id}")
    output_dir = "predictions" # Directory to save the plot
    os.makedirs(output_dir, exist_ok=True)
    
    # Data is already passed in
    df = combined_labels.copy()

    # Convert datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        try:
            df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
             print(f"Error converting datetime: {e}")
             return

    # Filter for the specific machine
    machine_df = df[df['machineID'] == machine_id].copy()

    if machine_df.empty:
        print(f"Error: No data found for Machine ID: {machine_id}")
        return

    # Create the combined color label if not already present (it should be)
    if 'ColorLabel' not in machine_df.columns:
        machine_df['ColorLabel'] = machine_df['LabelType'] + '_' + machine_df['LabelValue'].astype(str)
    
    # Sort by datetime for plotting
    machine_df = machine_df.sort_values('datetime')

    # --- Create Matplotlib Plot for the full series --- 
    print("Generating Matplotlib plot for full time series...")
    features_to_plot = ['volt', 'rotate', 'pressure', 'vibration']
    n_features = len(features_to_plot)
    
    fig, axes = plt.subplots(n_features, 1, figsize=(18, 10), sharex=True) # Wider figure for full series
    fig.suptitle(f"Full Telemetry Data for Machine ID {machine_id} Colored by Label Type/Value", fontsize=16)

    # --- Create Color Mapping --- 
    unique_labels = machine_df['ColorLabel'].unique()
    if len(unique_labels) <= 10:
        color_map = {label: mcolors.TABLEAU_COLORS[list(mcolors.TABLEAU_COLORS.keys())[i % len(mcolors.TABLEAU_COLORS)]] 
                     for i, label in enumerate(unique_labels)}
    else: # Fallback for more colors
        colors = plt.cm.get_cmap('viridis', len(unique_labels))
        color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    # --- Find GroundTruth_True event times for vertical lines --- 
    ground_truth_true_times = machine_df[machine_df['ColorLabel'] == 'GroundTruth_True']['datetime']

    # --- Plot each feature --- 
    all_handles = []
    all_labels = []
    legend_added_for_vlines = False # Flag to add vline legend entry only once
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        # Plot each group separately to assign colors and legend labels
        for label in unique_labels:
            subset = machine_df[machine_df['ColorLabel'] == label]
            # Only add handle/label to legend once
            if not subset.empty:
                handle = ax.scatter(subset['datetime'], subset[feature], 
                                   label=label, 
                                   color=color_map.get(label, '#808080'), # Use grey if label somehow missing from map
                                   s=5) # Smaller points for potentially dense full series
                if label not in all_labels:
                    all_handles.append(handle)
                    all_labels.append(label)
        
        ax.set_ylabel(feature.capitalize())
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add vertical lines for ALL GroundTruth_True events
        if not ground_truth_true_times.empty:
            for event_time in ground_truth_true_times:
                 ax.axvline(event_time, color='red', linestyle=':', linewidth=1.0)
            # Add legend entry for the vlines only once
            if not legend_added_for_vlines:
                line_handle = plt.Line2D([0], [0], color='red', linestyle=':', linewidth=1.0, label='GroundTruth_True Event')
                all_handles.append(line_handle)
                all_labels.append('GroundTruth_True Event')
                legend_added_for_vlines = True
            
        if i == n_features - 1: # Only bottom subplot needs x-label
            ax.set_xlabel("Time")
            ax.tick_params(axis='x', rotation=30)
            # Optionally format x-axis dates better for long series
            # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) 
            # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        else:
             ax.tick_params(axis='x', labelbottom=False) # Hide x-labels for upper plots

    # Adjust subplot parameters to make space for the legend on the right
    fig.subplots_adjust(right=0.85) # Reduce right boundary to make space

    # Add a single legend for the entire figure in the allocated space
    fig.legend(all_handles, all_labels, title="LabelType_LabelValue", 
               loc='center left', bbox_to_anchor=(0.87, 0.5)) # Place legend in the space created
    
    # Save the plot
    plot_filename = os.path.join(output_dir, f"full_series_viz_machine_{machine_id}.png")
    try:
        plt.savefig(plot_filename, dpi=300)
        print(f"Full series plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving full series plot: {e}")
    plt.close(fig) # Close the figure to free memory

def train():
    output_csv_dir = "data"
    only_maintenance_df = prepare_binary_classifier_trainset(only_maintenance=True, 
                                       lag_list=[1, 168],
                                       agg_list=['median', 'max', 'min', 'var'])
    only_maintenance_df.drop(columns=['daysSinceLastMaintenanceEvent'], inplace=True)
    train_df, test_df = split_data_by_date(only_maintenance_df)

    h2o.init(max_mem_size="2G")

    target = "isFailureEvent"   
    train_hf = h2o.H2OFrame(train_df)
    test_hf = h2o.H2OFrame(test_df)
    train_hf[target] = train_hf[target].asfactor()
    test_hf[target] = test_hf[target].asfactor()
    x = [col for col in train_hf.columns if col not in [target]]
    y = target

    aml = H2OAutoML(
        max_models=10,
        seed=42,
        max_runtime_secs=7200,
        sort_metric='accuracy'
    )
    aml.train(x=x, y=y, training_frame=train_hf)
    print(aml.leaderboard)

    # --- Save the Leader Model ---
    leader_model_path = "data/"
    os.makedirs(leader_model_path, exist_ok=True) # Ensure the directory exists
    model_path = h2o.save_model(model=aml.leader, path=leader_model_path, force=True)
    print(f"Leader model saved to: {model_path}")
    # -----------------------------

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
    
    # Accuracy extraction (handle potential list structure)
    train_acc = train_perf.accuracy()[0][1] if train_perf.accuracy() and len(train_perf.accuracy()) > 0 else np.nan
    test_acc = test_perf.accuracy()[0][1] if test_perf.accuracy() and len(test_perf.accuracy()) > 0 else np.nan
    print(f"Accuracy         | {train_acc:<9.4f} | {test_acc:<9.4f}")
    
    # Precision extraction (often corresponds to the threshold for max F1)
    # H2O returns a list of lists, typically [[threshold, precision]], we often want the second value if available
    train_prec = train_perf.precision()[0][1] if train_perf.precision() and len(train_perf.precision()) > 0 else np.nan
    test_prec = test_perf.precision()[0][1] if test_perf.precision() and len(test_perf.precision()) > 0 else np.nan
    print(f"Precision        | {train_prec:<9.4f} | {test_prec:<9.4f}")

    # Recall extraction (often corresponds to the threshold for max F1)
    train_rec = train_perf.recall()[0][1] if train_perf.recall() and len(train_perf.recall()) > 0 else np.nan
    test_rec = test_perf.recall()[0][1] if test_perf.recall() and len(test_perf.recall()) > 0 else np.nan
    print(f"Recall           | {train_rec:<9.4f} | {test_rec:<9.4f}")

    print("\n--- Confusion Matrix (Train) ---")
    print(train_perf.confusion_matrix())

    print("\n--- Confusion Matrix (Test) ---")
    print(test_perf.confusion_matrix())

    return leader

if __name__ == "__main__":
    model = train()
    preds = generate_preds(model, "") # If model trained here
    preds = generate_preds(None, "data/DeepLearning_1_AutoML_1_20240501_131549") # Load specific model
    contextualized_preds = contextualize_preds(None)
    visualize_preds(contextualized_preds, machine_id=1) # Visualize specific failure windows
    visualize_all_preds(contextualized_preds, machine_id=1) # Visualize full history