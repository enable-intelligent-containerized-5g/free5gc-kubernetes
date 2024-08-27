import csv
import os
import sys
import pandas as pd
import re

# Variables
dataset_path_default = dataset_path = "dataset"
data_path_default = data_path = 'data'

# Definir los nombres de las columnas, incluyendo la columna de fecha y hora
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
    "packets_receive", 
    "packets_transmit",
    "packets_total",
    "packets_receive_dropped", 
    "packets_transmit_dropped",
    "packets_total_dropped",
]

def input_dataset_path():
    input_dataset =  input(f"Ingresa el nombre del archivo CSV (sin la extensión). Presiona Enter para usar '{dataset_path_default}'): \n").strip()
    if input_dataset == "" and is_valid_name(dataset_path_default):
        return dataset_path_default
    elif is_valid_name(input_dataset):
        return  input_dataset
    else:
        print("Asegúrate de que el nombre solo contenga letras, números, guiones y guiones bajos, sin espacios.")
        return input_dataset_path()  # Vuelve a pedir el nombre

def input_data_path():
    input_data =  input(f"Introduce la ruta de los archivos de datos para crear el dataset. Presiona Enter para usar '{data_path_default}'): \n").strip()
    if input_data == "" and is_folder(data_path_default):
        return data_path_default
    elif is_folder(input_data):
        return input_data
    else:
        print("Asegúrate de que el directorio exista.")
        return input_data_path()  # Vuelve a pedir el nombre

def is_folder(folder):
    if os.path.isdir(folder):
        print(f"El directorio ingresado es '{folder}'\n.")
        return True
    else:
        print(f"La ruta '{folder}' no es un directorio.")
        return False

def is_valid_name(name):
    # La expresión regular verifica que el nombre solo contenga letras, números, guiones y guiones bajos
    pattern = r'^[\w-]+$'
    if bool(re.match(pattern, name)):
        print(f"El nombre ingresado es: '{name}'\n.")
        return True
    else:
        print(f"El nombre '{name}' no es un nombre valido.")
        return False

def exist_dataset(dataset_path):
    # Verifica si existe el archivo dataset
    if os.path.exists(dataset_path):
        return True
    else:
        return False
    
def read_csv_files(folder_path, dataset_df):
    # Asegurarse de que las columnas usadas para el índice sean del tipo `str`
    dataset_df['time'] = dataset_df['time'].astype(str)
    dataset_df['pod'] = dataset_df['pod'].astype(str)
    dataset_df['namespace'] = dataset_df['namespace'].astype(str)

    dataset_df = dataset_df.set_index(['time', 'pod', 'namespace'])

    # Obtener la lista de archivos en la carpeta
    files = os.listdir(folder_path)
    
    # Filtrar solo los archivos CSV
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Leer cada archivo CSV y procesar los datos
    for csv_file in csv_files:
        base_name = os.path.splitext(csv_file)[0]
        parts = base_name.split('_')
        
        if len(parts) == 5:
            namespace, nf, metric, date, time = parts[:5]
            file_path = os.path.join(folder_path, csv_file)
        
            try:
                df = pd.read_csv(file_path)
                
                if df.empty:
                    print(f"El archivo '{csv_file}' está vacío.\n")
                else:
                    # Asegurarse de que las columnas usadas para el índice sean del tipo `str`
                    df['time'] = df['time'].astype(str)
                    df['pod'] = nf
                    df['namespace'] = namespace
                    df['pod'] = df['pod'].astype(str)
                    df['namespace'] = df['namespace'].astype(str)

                    for index, row_df in df.iterrows():
                        new_row_df = pd.DataFrame([row_df])
                        new_row_df = new_row_df.set_index(['time', 'pod', 'namespace'])

                        # Verificar si las filas en new_row_df están en dataset_df
                        rows_missing = ~new_row_df.index.isin(dataset_df.index)
                        
                         # Verificar si la fila ya existe en dataset_df
                        if new_row_df.index.isin(dataset_df.index).any():
                            dataset_df.update(new_row_df)
                        else:
                            dataset_df = pd.concat([dataset_df, new_row_df], ignore_index=False)

            except pd.errors.EmptyDataError:
                print(f"El archivo '{csv_file}' no tiene datos o está vacío.\n")
        else:
            print(f"Nombre de archivo no esperado: {csv_file}\n")
    
    # Restaura el índice a columnas antes de guardar
    dataset_df = dataset_df.reset_index()

    # Guardar el dataset actualizado en el archivo CSV
    dataset_df.to_csv(dataset_path, index=False)
    print(f"El archivo '{dataset_path}' ha sido actualizado.")


dataset_path = input_dataset_path() + ".csv"
data_path = input_data_path()

if not exist_dataset(dataset_path):
    # Crear el archivo CSV y escribir la fila de encabezado
    with open(dataset_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
    print(f"File '{dataset_path}' created.\n")
else:
    print(f"El archivo '{dataset_path}' ya existe.")

# Leer el dataset.csv
dataset_df = pd.read_csv(dataset_path)

# Llamar a la función con la ruta de la carpeta 'data'
read_csv_files(data_path, dataset_df)
