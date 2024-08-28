import csv
import os
import sys
import pandas as pd
import re

# Variables
dataset_path_default = dataset_path = "dataset"
data_path_default = data_path = 'data'

# Set the columns
columns = [
    "time",
    "pod",
    "namespace",
    "cpu_request",
    "cpu_limit",
    "cpu_usage",
    "memory_request",
    "memory_limit",
    "memory_usage",
    "receive_packets",
    "transmit_packets",
    "total_packets",
    "receive_packets_dropped",
    "transmit_packets_dropped",
    "total_packets_dropped"
]

def yes_no_question(input_ms, yes_ms, not_ms):
    user_input = input(input_ms).strip().lower()
    if user_input == 'y':
        # Use default values
        print(yes_ms)
        return True
    elif user_input == 'n':
        print(not_ms)
        return False
    else:
        print("Error: Invalid input. Please enter 'y' or 'n'.")
        return yes_no_question(input_ms, yes_ms, not_ms)

def duplicated_indexes(data_frame):
    # Find all duplicate indices, including the first occurrence
    return data_frame.index.duplicated(keep='first')

def input_dataset_path():
    input_dataset =  input(f"Enter the CSV file name (without the extension) or press Enter to use the default '{dataset_path_default}'): ").strip()
    if input_dataset == "" and is_valid_name(dataset_path_default):
        return dataset_path_default
    elif is_valid_name(input_dataset):
        return  input_dataset
    else:
        print("Error: Make sure the name contains only letters, numbers, hyphens, and underscores, with no spaces.")
        return input_dataset_path()

def input_data_path():
    input_data =  input(f"Enter the path to the data files to create the dataset. Press Enter to use the default. '{data_path_default}'): ").strip()
    if input_data == "" and is_folder(data_path_default):
        return data_path_default
    elif is_folder(input_data):
        return input_data
    else:
        print(f"Error: Make sure the directory '{input_data}' exists.")
        return input_data_path()

def is_folder(folder):
    if os.path.isdir(folder):
        print(f"The entered directory is '{folder}'.")
        return True
    else:
        print(f"Error: The path '{folder}' isn't a directory.")
        return False

def is_valid_name(name):
    # The regular expression checks that the name contains only letters, numbers, hyphens, and underscores.
    pattern = r'^[\w-]+$'
    if bool(re.match(pattern, name)):
        print(f"The name entered is '{name}'.")
        return True
    else:
        print(f"Error: The name '{name}' isn't a valid name.")
        return False

def exist_dataset(dataset_path):
    # Check if the dataset file exists.
    if os.path.exists(dataset_path):
        return True
    else:
        return False
    
def update_dataset(df1, df2, file_name):
    # Existing rows in `df2`
    rows_existing = df1.index.isin(df2.index)
    rows_missing = ~rows_existing
    
    # Update and Add the new rows 
    df2.update(df1[rows_existing])
    df2 = pd.concat([df2, df1[rows_missing]], ignore_index=False)

    print(f"Success: File '{file_name}' added.\n")
    return df2

    
def read_csv_files(folder_path, dataset_df):
    # set  the 'time', 'pod' y 'namespace' columns like index.
    dataset_df = dataset_df.set_index(['time', 'pod', 'namespace'])

    # Get the CSV files.
    files = os.listdir(folder_path)
    
    # Filter only the CSV files.
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Read each CSV file and process the data.
    for count, csv_file  in enumerate(csv_files):
        print(f"PROCESSING FILE #{count + 1}: '{csv_file}'.")

        base_name = os.path.splitext(csv_file)[0]
        parts = base_name.split('_')
        
        if len(parts) == 5:
            namespace, nf, metric, date, time = parts[:5]
            file_path = os.path.join(folder_path, csv_file)
        
            try:
                df = pd.read_csv(file_path)

                # Add the nf and namespace columns.
                if 'pod' not in df.columns:
                    df['pod'] = nf
                if 'namespace' not in df.columns:
                    df['namespace'] = namespace

                # set  the 'time', 'pod' y 'namespace' columns like index
                df = df.set_index(['time', 'pod', 'namespace'])

                # Duplicated indexes
                duplicated_indexes_df = duplicated_indexes(df)
                duplicated_df = df.index[duplicated_indexes_df]
                num_duplicate_indexes = len(duplicated_df)
                
                if df.empty:
                    print(f"Error: The file '{csv_file}' is empty.\n")

                elif duplicated_indexes_df.any():
                    print(f"Error: {num_duplicate_indexes} Duplicated indexes found in '{csv_file}' \n-> {duplicated_df}")
                    removing_idx_dp = yes_no_question(f"Do you want to remove duplicate indices from '{csv_file}' and add them to the dataset? (y/n): ", "Removing duplicate indices", f"File '{csv_file}' discarded")
                    if removing_idx_dp:
                        df = df[~duplicated_indexes_df]
                        dataset_df = update_dataset(df, dataset_df, csv_file)

                        # Save the df
                        df = df.reset_index()
                        df.to_csv(data_path + "/" + csv_file, index=False)
                    else:
                        print()
                
                else:    
                    dataset_df = update_dataset(df, dataset_df, csv_file)

            except pd.errors.EmptyDataError:
                print(f"Error: The file '{csv_file}' haven't data or is empty.\n")
        else:
            print(f"Error: Name file '{csv_file}' unexpected. Required name: 'namespace_nf_metric_date_time.csv'.\n")
    
    # Reset the column indexes
    dataset_df = dataset_df.reset_index()

    # Save the data in a CSV file
    dataset_df.to_csv(dataset_path, index=False)
    print(f"The file '{dataset_path}' has been updated.")

print(f"ENTER THE 'dataset' and 'data' PATH.")
if not yes_no_question("Do you want to use the default values? (y/n): ", "Using default values.\n", "Enter the custom values:"):
    dataset_path = input_dataset_path() + ".csv"
    data_path = input_data_path()
else:
    dataset_path = dataset_path_default + ".csv"
    data_path = data_path_default

if not exist_dataset(dataset_path):
    # Create the CSV file and write the head row
    with open(dataset_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
    print(f"File '{dataset_path}' created.\n")
else:
    print(f"The dataset '{dataset_path}' will be overwritten.\n")

# Read the dataset.csv
dataset_df = pd.read_csv(dataset_path)

# Read the csv files
read_csv_files(data_path, dataset_df)
