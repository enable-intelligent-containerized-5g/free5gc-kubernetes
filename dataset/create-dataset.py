import csv
import os
import pandas as pd

# Variables
dataset_path = "dataset.csv"
data_folder = 'data'

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

# Verifica si existe el archivo dataset
if not os.path.exists(dataset_path):
    # Crear el archivo CSV y escribir la fila de encabezado
    with open(dataset_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)
    print(f"File '{dataset_path}' created.\n")

# Leer el dataset.csv
dataset_df = pd.read_csv(dataset_path)

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
        parts = base_name.split('.')
        
        # Verificar que el archivo tenga al menos 4 partes
        if len(parts) == 4:
            metric, nf, namespace, time = parts[:4]
            print(f"Archivo: {csv_file}")
            print(f"Metric: {metric}, NF: {nf}, Namespace: {namespace}, Time: {time}")

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
                    for index, row in df.iterrows():
                        time_value = row['time']
                        
                        # Verificar si el valor de Time no está en la columna 'time' del dataset
                        if time_value not in existing_times:
                            # Agrega las nuevas claves al diccionario
                            row['pod'] = nf
                            row['namespace'] = namespace
                            
                            # Crear un DataFrame con una sola fila y concatenarlo con el dataset
                            new_row_df = pd.DataFrame([row])
                            dataset_df = pd.concat([dataset_df, new_row_df], ignore_index=True)
                            print(f"Fila con Time '{time_value}' añadida desde '{csv_file}'.")
                        else:
                            print(f"Fila con Time '{time_value}' ya existe en el valor")
                    
                    print()
        
            except pd.errors.EmptyDataError:
                print(f"El archivo '{csv_file}' no tiene datos o está vacío.\n")
            
        else:
            print(f"Nombre de archivo no esperado: {csv_file}\n")
    
    # Guardar el dataset actualizado en el archivo CSV
    dataset_df.to_csv(dataset_path, index=False)
    print(f"El archivo '{dataset_path}' ha sido actualizado.")

# Llamar a la función con la ruta de la carpeta 'data'
read_csv_files(data_folder, dataset_df)
