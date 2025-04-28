import pandas as pd
import numpy as np
import os
import h2o
from h2o.automl import H2OAutoML
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from data_utilities import prepare_data, split_data_by_date

def prepare_feature_target(df, target_col):
    # Drop target, machineID, datetime

    cols_to_drop = [target_col, 'datetime', 'machineID']
    X = df.drop(columns=cols_to_drop, errors='ignore')

    # Target
    y = df[target_col]

    return X, y

def evaluate_model(model, X_test, y_test):
    """Evaluates the model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except Exception:
        roc_auc = None
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'report': report
    }
    
    return metrics

def plot_correlation_heatmap(df, title="Feature Correlation Heatmap", save_path=None):
    """Calculates and plots the correlation matrix for numeric columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        title (str): Title for the heatmap plot.
        save_path (str, optional): Path to save the heatmap image. If None, displays the plot.
    """
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=np.number)
    
    if numeric_df.empty:
        print("No numeric columns found to calculate correlation.")
        return
        
    print(f"Calculating correlation matrix for {len(numeric_df.columns)} numeric features...")
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=(18, 15)) # Adjust size as needed
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f") # annot=True can be slow for many features
    plt.title(title, fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300)
            print(f"Correlation heatmap saved to {save_path}")
        except Exception as e:
            print(f"Error saving heatmap: {e}")
        plt.close() # Close the plot figure after saving
    else:
        plt.show()

if __name__ == "__main__":
    only_maintenance_df = prepare_data(only_maintenance=True, 
                                       lag_list=[1, 12, 72, 168],
                                       agg_list=['median', 'max', 'min', 'var'])
    only_maintenance_df.drop(columns=['daysSinceLastMaintenanceEvent'], inplace=True)
    train_df, test_df = split_data_by_date(only_maintenance_df)

    # Prepare features and target
    target = "isFailureEvent"
    X_train, y_train = prepare_feature_target(train_df, target)
    X_test, y_test = prepare_feature_target(test_df, target)

    # --- Check and Convert Target Variable --- 
    print(f"\nTarget variable type before conversion: {y_train.dtype}")
    print(f"Unique values in y_train before conversion: {y_train.unique()}")
    print(f"NaNs in y_train: {y_train.isnull().sum()}")
    print(f"NaNs in y_test: {y_test.isnull().sum()}")

    # Handle potential NaNs (e.g., fill with the mode or drop rows)
    # For simplicity, let's fill with 0 (assuming False is the majority/default)
    # A more robust approach might be needed depending on the data
    y_train = y_train.fillna(0)
    y_test = y_test.fillna(0)
    
    # Convert target to integer type
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    print(f"Target variable type after conversion: {y_train.dtype}")
    print(f"Unique values in y_train after conversion: {y_train.unique()}")
    # ----------------------------------------

    print(f"Training features shape before preprocessing: {X_train.shape}")
    print(f"Testing features shape before preprocessing: {X_test.shape}")

    # --- Plot Correlation Heatmap --- 
    result_dir = "data"
    plot_correlation_heatmap(X_train, title="Training Feature Correlation Heatmap", save_path=f"plots/feature_correlation_heatmap.png")
    # --------------------------------
    
    # Identify feature types
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure 'model' is treated as categorical if present
    if 'model' in X_train.columns and 'model' not in categorical_features:
         categorical_features.append('model')
    if 'model' in numeric_features: # Remove model from numeric if misclassified
        numeric_features.remove('model')


    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")


    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handle potential missing categories
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # Use OneHotEncoder
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features) # Add categorical transformer
        ],
        remainder='drop' # Keep other columns if any (though should be none with current logic)
    )

    # Build model pipeline with logistic regression
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=1.0,
            penalty='l2',
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear',
            random_state=42
        ))
    ])

    # Train the model
    print("\nTraining logistic regression model...")
    model.fit(X_train, y_train)

    # Get feature names after preprocessing (important for interpretation)
    try:
        # Get feature names from numeric transformer
        num_feature_names = numeric_features
        # Get feature names from categorical transformer (OneHotEncoder)
        cat_feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
        # Combine feature names
        processed_feature_names = num_feature_names + cat_feature_names
    except Exception as e:
        print(f"Could not get feature names: {e}")
        # Fallback if getting names fails
        processed_feature_names = [f'feature_{i}' for i in range(model.named_steps['classifier'].coef_.shape[1])]


    # Model coefficients
    if len(processed_feature_names) == model.named_steps['classifier'].coef_.shape[1]:
        coefficients = model.named_steps['classifier'].coef_[0]

        # Create a DataFrame of feature importance
        feature_importance = pd.DataFrame({
            'Feature': processed_feature_names,
            'Coefficient': coefficients
        })

        # Sort by absolute coefficient value
        feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
        feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

        # Save feature importance
        result_dir = "data"
        os.makedirs(result_dir, exist_ok=True)
        feature_importance.to_csv(f"{result_dir}/logistic_regression_feature_importance.csv", index=False)
    else:
        print("\nWarning: Mismatch between number of feature names and coefficients. Cannot display feature importance.")
        print(f"Number of names: {len(processed_feature_names)}, Number of coefficients: {model.named_steps['classifier'].coef_.shape[1]}")


    # Evaluate model
    print("\nEvaluating model on test data...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nLogistic Regression Performance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}" if metrics['roc_auc'] else "ROC AUC: N/A")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"True Negatives: {metrics['true_negatives']}")
    print(f"False Positives: {metrics['false_positives']}")
    print(f"False Negatives: {metrics['false_negatives']}")
    print(f"True Positives: {metrics['true_positives']}")
    
    # Save model predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    test_results = test_df.copy()
    test_results['predicted_failure'] = y_pred
    test_results['failure_probability'] = y_pred_proba

    test_results.to_csv(f"{result_dir}/logistic_regression_predictions.csv", index=False)

    print(f"\nResults saved to {result_dir} directory")


