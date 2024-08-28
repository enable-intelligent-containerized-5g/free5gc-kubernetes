import os
import json
import pandas as pd

# Directorio donde están los archivos JSON
json_directory = 'result-querys'

# Crear un DataFrame final vacío
df_final = pd.DataFrame()

# Iterar sobre cada archivo en el directorio
for filename in os.listdir(json_directory):
    print(f"FILE: '{filename}'")
    if filename.endswith('.json'):
        # Parsear el nombre del archivo para extraer el namespace y la métrica
        namespace, metric, date, time = filename.split('_')
        
        # Leer el archivo JSON
        with open(os.path.join(json_directory, filename), 'r') as file:
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
            
            # Crear un DataFrame temporal con las filas
            temp_df = pd.DataFrame(rows)
            
            # Configurar el índice en el DataFrame temporal
            temp_df.set_index(['time', 'namespace', 'pod', 'container'], inplace=True)
            
            # Actualizar el DataFrame final con los datos del DataFrame temporal
            df_final = pd.concat([df_final, temp_df], axis=1, join='outer')

# Guardar el DataFrame final como un CSV
df_final.to_csv('output.csv')

print("CSV creado exitosamente con los datos agregados o actualizados en función del índice.")