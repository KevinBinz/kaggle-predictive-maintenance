import pytest
import pandas as pd
import os
import sys

# Add the parent directory (features) to the Python path
# This allows importing from features.data_preparation
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from features.data_preparation import process_data

@pytest.fixture(scope='module')
def final_data():
    """
    Fixture to LOAD the pre-computed final data from the CSV file.
    Ensures data preparation pipeline is run once outside the test.
    """
    final_data_path = os.path.join(parent_dir, 'data', 'final.csv') # Construct path relative to project root
    print(f"\n>>> Loading final data for tests from: {final_data_path}")
    
    try:
        # Ensure the file exists before trying to load
        if not os.path.exists(final_data_path):
             pytest.fail(f"Final data file not found at {final_data_path}. Run data preparation script first.", pytrace=False)
             
        final_df = pd.read_csv(final_data_path)
        # Ensure datetime is in the correct format for tests
        final_df['datetime'] = pd.to_datetime(final_df['datetime'])
        print("Data loaded successfully for tests.")
        return final_df
    except Exception as e:
        pytest.fail(f"Error loading final data from {final_data_path}: {e}", pytrace=False)

def test_final_df_unique_datetime_per_machine(final_data):
    """
    Test 1: Asserts that each combination of machineID and datetime is unique
    in the final processed DataFrame.
    """
    duplicates = final_data[final_data.duplicated(subset=['machineID', 'datetime'], keep=False)]
    num_duplicates = len(duplicates)
    assert num_duplicates == 0, (
        f"Found {num_duplicates} duplicate (machineID, datetime) combinations. "
        f"Example duplicates:\n{duplicates.head()}"
    )
    # assert True

def test_final_df_hourly_completeness_2015(final_data):
    """
    Test 2: Asserts that there is exactly one record for every hour
    for every machine throughout the year 2015.
    """
    # Filter for the year 2015
    df_2015 = final_data[final_data['datetime'].dt.year == 2015].copy()
    # Check if any data exists for 2015
    assert not df_2015.empty, "No data found for the year 2015 to test completeness."
    # Calculate expected hours in 2015 (not a leap year)
    expected_hours_per_machine = 365 * 24 - 6
    # Group by machineID and count records
    counts_per_machine = df_2015.groupby('machineID').size()
    # Check if all machines are present in the counts
    all_machines = final_data['machineID'].unique()
    machines_in_2015 = counts_per_machine.index.unique()
    missing_machines = set(all_machines) - set(machines_in_2015)
    assert not missing_machines, f"Machines missing entirely from 2015 data: {missing_machines}"
    # Check if each machine has the expected number of records
    incorrect_counts = counts_per_machine[counts_per_machine != expected_hours_per_machine]
    assert incorrect_counts.empty, (
        f"Found machines with incorrect number of hourly records in 2015 "
        f"(expected {expected_hours_per_machine}):\n{incorrect_counts.to_string()}"
    )
    # assert True

# def test_collection(): # Comment out or remove dummy test
#     """Dummy test to see if collection works at all."""
#     print("\n>>> test_collection running")
#     assert True
