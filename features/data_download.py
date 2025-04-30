import kagglehub
import os
import glob
import shutil

"""
To run this script, you will first need to authenticate using an API token. 
https://www.kaggle.com/docs/api#authentication
"""

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")
print("Path to downloaded dataset files:", path)

# --- Move CSV files to ./telemetry/ --- 
target_dir = "telemetry"
print(f"\nMoving CSV files from '{path}' to '{target_dir}'...")

# Ensure the target directory exists
try:
    os.makedirs(target_dir, exist_ok=True)
except OSError as e:
    print(f"Error creating directory {target_dir}: {e}")
    # Decide how to handle this error, e.g., exit
    exit(1) 

# Find all CSV files in the downloaded path
csv_pattern = os.path.join(path, "*.csv")
csv_files = glob.glob(csv_pattern)

if not csv_files:
    print(f"Warning: No CSV files found in {path}")
else:
    print(f"Found {len(csv_files)} CSV files to move.")
    moved_count = 0
    for file_path in csv_files:
        try:
            filename = os.path.basename(file_path)
            dest_path = os.path.join(target_dir, filename)
            print(f"  Moving '{filename}' to '{dest_path}'")
            shutil.move(file_path, dest_path)
            moved_count += 1
        except Exception as e:
            print(f"  Error moving file {filename}: {e}")
            
    print(f"Successfully moved {moved_count} CSV files to {target_dir}.")
# ------------------------------------