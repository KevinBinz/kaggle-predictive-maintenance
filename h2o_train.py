import pandas as pd
import numpy as np
import os
import h2o
from h2o.automl import H2OAutoML
from data_utilities import split_data_by_date, prepare_data


if __name__ == "__main__":
    output_csv_dir = "data"
    only_maintenance_df = prepare_data(only_maintenance=True, 
                                       lag_list=[1, 168],
                                       agg_list=['median', 'max', 'min', 'var'])
    only_maintenance_df.drop(columns=['daysSinceLastMaintenanceEvent'], inplace=True)
    train_df, test_df = split_data_by_date(only_maintenance_df)

    h2o.init()

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
    # Accuracy needs careful indexing as it might be nested
    train_acc = train_perf.accuracy()[0][1] if train_perf.accuracy() and len(train_perf.accuracy()) > 0 else np.nan
    test_acc = test_perf.accuracy()[0][1] if test_perf.accuracy() and len(test_perf.accuracy()) > 0 else np.nan
    print(f"Accuracy         | {train_acc:<9.4f} | {test_acc:<9.4f}")

    print("\n--- Confusion Matrix (Train) ---")
    print(train_perf.confusion_matrix())

    print("\n--- Confusion Matrix (Test) ---")
    print(test_perf.confusion_matrix())
