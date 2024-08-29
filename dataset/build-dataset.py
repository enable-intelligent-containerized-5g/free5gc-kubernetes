import os
import json
import csv
import argparse
import pandas as pd

columns = ["time", "namespace", "pod", "container"]

indexes = ['time', 'namespace', 'pod', 'container']

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

    print(f"  Success: File '{file_name}' added.\n")
    return df2
    
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

def load_container_if_not_exist(pod_data_path, metric_info, pod):
    if not 'container' in metric_info:
        print("  Warning: Container property not in metric info, loading it from pod list...")
        with open(pod_data_path, 'r') as file:
            json_pod_data = json.load(file)
            if pod in json_pod_data["pods"]:
                value = json_pod_data["pods"][pod][0]
                print(f"    Using '{pod}': {value}, from {pod_data_path}.")
                return value
            else:
                print(f"  Error: Key '{pod}' not found in {pod_data_path}. Container name not loaded")
    else:
        return metric_info['container']

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Build the dataset.')

    # Add args
    default_output = "dataset.csv"
    default_data = "data/"
    parser.add_argument('-o', '--output', dest='output', type=str, required=False, 
                        help=f"Dataset directory, default is '{default_output}'.", 
                        default=default_output)
    parser.add_argument('-d', '--data', dest='data', type=str, required=False, 
                        help=f"Data directory, default is '{default_data}'.", 
                        default=default_data)

    # Parse the args
    args = parser.parse_args()
    arg_output = args.output
    arg_data = args.data.rstrip('/')
    pod_data_path = f"{arg_data}/pod-data.json"

    if not exist_dataset(arg_output):
        # Create the CSV file and write the head row
        with open(arg_output, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
        print(f"File '{arg_output}' created.\n")
    else:
        print(f"The dataset '{arg_output}' will be overwritten.\n")

    df_final = pd.read_csv(arg_output)
    df_final.set_index(indexes)

    # Filter only the JSON files.
    files = os.listdir(arg_data)
    files = [f for f in files if f.endswith('.json')]

    # Read each JSON file and process the data.
    for count_files, filename in enumerate(files):
        print(f"PROCESSING FILE #{count_files + 1}: '{filename}'")
        base_filename = os.path.splitext(filename)[0]
        parts_filename = base_filename.split('_')

        if len(parts_filename) == 5:
            if filename.endswith('.json'):
                # Parse the filename and extract the namespace and the metric
                namespace, metric = filename.split('_')[:2]

                # Add metric column
                if metric not in df_final.columns:
                    df_final[metric] = pd.NA
                
                # Read the JSON file
                with open(os.path.join(arg_data, filename), 'r') as file:
                    json_data = json.load(file)
                
                # Verify the JSON data
                if (json_data.get('status') == 'success') and json_data['data']['result']:
                    # Extract the result
                    results = json_data['data']['result']
                    
                    """List for temporal DataFrame"""
                    rows = []
                    
                    """Extract the pod, container, time, and value"""
                    for result in results:
                        metric_info = result['metric']
                        if 'pod' in metric_info:
                            pod = metric_info['pod']
                            container = load_container_if_not_exist(pod_data_path, metric_info, pod)
                            values = result['values']
                            """Add value to rows"""
                            for value in values:
                                timestamp, metric_value = value
                                """Create a row in the temporal DataFrame"""
                                rows.append({
                                    'time': timestamp,
                                    'namespace': namespace,
                                    'pod': pod,
                                    'container': container,
                                    metric: metric_value
                                })
                        else:
                            print(f"  Error: The pod property is not in the metric property of the '{filename}' file.")
                    
                    # Create a temporal DataFrame
                    temp_df = pd.DataFrame(rows)
                    
                    # Set the index to temporal DataFrame
                    if not temp_df.empty:
                        temp_df.set_index(indexes)

                    # Duplicated indexes
                    duplicated_indexes_temp_df = duplicated_indexes(temp_df)
                    duplicated_temp_df = temp_df.index[duplicated_indexes_temp_df]
                    num_duplicate_indexes_temp_df = len(duplicated_temp_df)
                    
                    if temp_df.empty:
                        print(f"  Error: The file '{filename}' is empty.\n")

                    elif duplicated_indexes_temp_df.any():
                        print(f"  Error: {num_duplicate_indexes_temp_df} Duplicated indexes found in '{filename}' \n-> {duplicated_temp_df}")
                        removing_idx_dp = yes_no_question(
                            f"  Do you want to remove duplicate indices from '{filename}' and add them to the dataset? (y/n): ", 
                            "Removing duplicate indices", 
                            f"File '{filename}' discarded")
                        if removing_idx_dp:
                            temp_df = temp_df[~duplicated_indexes_temp_df]
                            df_final = update_dataset(temp_df, df_final, filename)

                            # Save the temp_df
                            temp_df = temp_df.reset_index()
                            temp_df.to_csv(arg_data + "/" + filename, index=False)
                        else:
                            print()
                    else:
                        df_final = update_dataset(temp_df, df_final, filename)
                else:
                    print(f"  Error: The file '{filename}' haven't data or is empty.\n")
            else:
                print(f"  Error: Name file '{filename}' isn't JSON file.\n")
        else:
            print(f"  Error: Name file '{filename}' unexpected. Required name: 'namespace_metric_duration_date_time.json'.\n")

    # Reset the column indexes
    df_final = df_final.reset_index(drop=True)

    # Delete row with null or undefined columns and Save the data in a CSV file
    df_final = df_final.dropna()
    df_final = df_final[~df_final.astype(str).apply(lambda x: x.str.contains('undefined', na=False)).any(axis=1)]
    
    # Separate the columns to exclude from the sorting
    excluded_df = df_final[indexes]

    # Sort the remaining columns alphabetically
    df_final = df_final[[col for col in df_final.columns if col not in indexes]]
    df_final = df_final[sorted(df_final.columns)]

    # Combine the excluded and sorted columns
    df_final = pd.concat([excluded_df, df_final], axis=1)

    # Save the dataset like a CSV file
    df_final.to_csv(arg_output, index=False)
    print(f"The file '{arg_output}' has been updated.")

if __name__ == '__main__':
    main()