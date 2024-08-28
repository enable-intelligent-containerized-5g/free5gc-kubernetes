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

    print(f"Success: File '{file_name}' added.\n")
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

def main():
    # Create the parser
    parcer = argparse.ArgumentParser(description='Descripción de tu script.')

    # Add args
    default_output = "dataset.csv"
    default_data = "data/"
    parcer.add_argument('-o', '--output', dest='output', type=str, required=False, 
                        help=f'Dataset directory (default is "{default_output}").', 
                        default=default_output)
    parcer.add_argument('-d', '--data', dest='data', type=str, required=False, 
                        help=f'Data directory (default is "{default_data}").', 
                        default=default_data)

    # Parsea los argumentos
    args = parcer.parse_args()
    args_output = args.output
    args_data = args.data.rstrip('/')

    if not exist_dataset(args_output):
        # Create the CSV file and write the head row
        with open(args_output, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
        print(f"File '{args_output}' created.\n")
    else:
        print(f"The dataset '{args_output}' will be overwritten.\n")

    df_final = pd.read_csv(args_output)
    df_final.set_index(indexes)

    # Filter only the CSV files.
    files = os.listdir(args_data)
    files = [f for f in files if f.endswith('.json')]

    # Read each JSON file and process the data.
    for count_files, filename in enumerate(files):
        print(f"PROCESSING FILE #{count_files + 1}: '{filename}'")
        base_filename = os.path.splitext(filename)[0]
        parts_filename = base_filename.split('_')

        if len(parts_filename) == 4:
            if filename.endswith('.json'):
                # Parsear el nombre del archivo para extraer el namespace y la métrica
                namespace, metric, date, time = filename.split('_')

                # Add metric column
                if metric not in df_final.columns:
                    df_final[metric] = pd.NA
                
                # Leer el archivo JSON
                with open(os.path.join(args_data, filename), 'r') as file:
                    json_data = json.load(file)
                
                # Verificar que el JSON tiene un estado de éxito
                if json_data.get('status') == 'success':
                    # Extraer los resultados
                    results = json_data['data']['result']
                    
                    # Crear una lista de filas para el DataFrame temporal
                    rows = []
                    
                    # Procesar cada resultado para extraer el pod, container, tiempo y valor
                    for result in results:
                        metric_info = result['metric']
                        if 'container' in metric_info:
                            pod = metric_info['pod']
                            container = metric_info['container']
                            values = result['values']
                            
                            # Agregar cada valor a la lista de filas
                            for value in values:
                                timestamp, metric_value = value
                                # Crear una fila en el DataFrame temporal
                                rows.append({
                                    'time': timestamp,
                                    'namespace': namespace,
                                    'pod': pod,
                                    'container': container,
                                    metric: metric_value
                                })
                    
                    # Create a temporal DataFrame
                    temp_df = pd.DataFrame(rows)
                    
                    # Set the index to temporal DataFrame
                    temp_df.set_index(indexes)

                    # OPTION 1
                    # Duplicated indexes
                    duplicated_indexes_temp_df = duplicated_indexes(temp_df)
                    duplicated_temp_df = temp_df.index[duplicated_indexes_temp_df]
                    num_duplicate_indexes_temp_df = len(duplicated_temp_df)
                    
                    if temp_df.empty:
                        print(f"Error: The file '{filename}' is empty.\n")

                    elif duplicated_indexes_temp_df.any():
                        print(f"Error: {num_duplicate_indexes_temp_df} Duplicated indexes found in '{filename}' \n-> {duplicated_temp_df}")
                        removing_idx_dp = yes_no_question(
                            f"Do you want to remove duplicate indices from '{filename}' and add them to the dataset? (y/n): ", 
                            "Removing duplicate indices", 
                            f"File '{filename}' discarded")
                        if removing_idx_dp:
                            temp_df = temp_df[~duplicated_indexes_temp_df]
                            df_final = update_dataset(temp_df, df_final, filename)

                            # Save the temp_df
                            temp_df = temp_df.reset_index()
                            temp_df.to_csv(args_data + "/" + filename, index=False)
                        else:
                            print()
                    
                    else:
                        df_final = update_dataset(temp_df, df_final, filename)
                        
        else:
            print(f"Error: Name file '{filename}' unexpected. Required name: 'namespace_metric_date_time.json'.\n")

    # Reset the column indexes
    df_final = df_final.reset_index(drop=True)

    # Delete row with null or undefined columns and Save the data in a CSV file
    df_final = df_final.dropna()
    df_final = df_final[~df_final.astype(str).apply(lambda x: x.str.contains('undefined', na=False)).any(axis=1)]
    
    # Save the dataset like a CSV file
    df_final.to_csv(args_output, index=False)
    print(f"The file '{args_output}' has been updated.")

if __name__ == '__main__':
    main()