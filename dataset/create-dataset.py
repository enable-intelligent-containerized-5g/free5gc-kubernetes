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
        return dataset_path_default + ".csv"
    elif is_valid_name(input_dataset):
        return  input_dataset + ".csv"
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

def exist_dataset():
    # Verifica si existe el archivo dataset
    if not os.path.exists(dataset_path):
        return True
    else:
        return False
    
def read_csv_files(folder_path, dataset_df):
    # Obtener la columna 'time' del dataset existente
    if 'time' in dataset_df.columns:
        existing_times = dataset_df['time'].tolist()
    else:
        existing_times = []

    # Obtener la lista de archivos en la carpeta
    files = os.listdir(folder_path)
    
    # Filtrar solo los archivos CSV
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Leer cada archivo CSV y procesar los datos
    for csv_file in csv_files:
        # Extraer el nombre base del archivo sin la extensión
        base_name = os.path.splitext(csv_file)[0]
        # Dividir el nombre base en partes usando puntos como separadores
        parts = base_name.split('_')
        
        # Verificar que el archivo tenga al menos 4 partes
        if len(parts) == 5:
            namespace, nf, metric, date, time = parts[:5]
            print(f"Archivo: {csv_file}")
            print(f"Namespace: {namespace}, NF: {nf}, Metric: {metric}, Date: {date}, Time: {time}")

            file_path = os.path.join(folder_path, csv_file)
        
            try:
                # Leer el archivo CSV usando pandas
                df = pd.read_csv(file_path)
                
                # Verificar si el DataFrame está vacío
                if df.empty:
                    print(f"El archivo '{csv_file}' está vacío.\n")
                else:
                    # Mostrar las primeras filas del DataFrame
                    print(f"Contenido de {csv_file}:")
                    print(df.head(3))
                    print()  # Imprimir una línea en blanco

                    # Recorrer cada fila del archivo CSV
                    for index, row_df in df.iterrows():
                        # Agrega las nuevas claves al diccionario
                        row_df['pod'] = nf
                        row_df['namespace'] = namespace
                        
                        # Time value
                        time_value = row_df['time']

                        # Crear un DataFrame con una sola fila
                        new_row_df = pd.DataFrame([row_df])
                        # Is time_value present
                        is_time_present = time_value in existing_times
                        
                        # Verificar si el valor de Time no está en la columna 'time' del dataset
                        if not is_time_present:                
                            # Concatena el DataFrame new_row_df con el dataset
                            print("not present")
                            dataset_df = pd.concat([dataset_df, new_row_df], ignore_index=True)
                            # print(f"Fila con Time '{time_value}' añadida desde '{csv_file}'.")
                        else:
                            print("Present")
                            rows_found_dataset = dataset_df[dataset_df['time'] == time_value]
                            print(f"Found: {rows_found_dataset}")

                            for index, row_found in rows_found_dataset.iterrows():
                                pod_dataset = row_found['pod']
                                ns_dataset = row_found['namespace']
                                
                                # Comparar los valores de pod y namespace con los valores especificados
                                if (pod_dataset == nf) and (ns_dataset == namespace):
                                    print(f"Ya existe Fila con (time, pod, namespace) = {time_value, pod_dataset, ns_dataset}'")
                                    # print(f"Fila con (time, pod, namespace) = {time_value, pod_dataset, ns_dataset}' ya existe en el dataset")
                                else:
                                    dataset_df = dataset_df[~(
                                        (dataset_df['time'] == time_value) &
                                        (dataset_df['pod'] == nf) &
                                        (dataset_df['namespace'] == namespace)
                                    )]
                                    dataset_df = pd.concat([dataset_df, new_row_df], ignore_index=True)
                                    print(f"Update Fila con (time, pod, namespace) = {time_value, pod_dataset, ns_dataset}' desde '{csv_file}'.")

                    print()
        
            except pd.errors.EmptyDataError:
                print(f"El archivo '{csv_file}' no tiene datos o está vacío.\n")
            
        else:
            print(f"Nombre de archivo no esperado: {csv_file}\n")
    
    # Guardar el dataset actualizado en el archivo CSV
    dataset_df.to_csv(dataset_path, index=False)
    print(f"El archivo '{dataset_path}' ha sido actualizado.")

dataset_path = input_dataset_path()
data_path = input_data_path()
if exist_dataset():
    # Crear el archivo CSV y escribir la fila de encabezado
    with open(dataset_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
    print(f"File '{dataset_path}' created.\n")
else:
    print(f"El archivo '{dataset_path} no existe.")
    sys.exit()

# Leer el dataset.csv
dataset_df = pd.read_csv(dataset_path)

# Llamar a la función con la ruta de la carpeta 'data'
read_csv_files(data_path, dataset_df)
