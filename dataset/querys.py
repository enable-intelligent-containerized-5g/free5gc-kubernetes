import requests
import json
from datetime import datetime, timezone, timedelta
import re
import subprocess
import argparse

result_querys = "result-querys/"


def parse_duration(duration_str):
    """Convierte una cadena de duración (ej. '1h', '30m') a segundos"""
    unit = duration_str[-1].lower()
    value = int(duration_str[:-1])
    if unit == 'h':
        return value * 3600
    elif unit == 'm':
        return value * 60
    else:
        raise ValueError(
            "Formato de duración no válido. Use 'h' para horas o 'm' para minutos.")


def get_epoch_range(duration):
    """Calcula el rango de tiempo en epoch basado en la duración"""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(seconds=duration)
    return int(start_time.timestamp()), int(end_time.timestamp())


def generate_filename(namespace, resource, timestamp):
    """Genera un nombre de archivo basado en el namespace y el recurso"""
    return f"{namespace}_{resource}_{timestamp}.json"


def fetch_and_save_query(base_url, params, query_info, namespace):
    """Realiza la consulta y guarda los resultados en un archivo JSON"""
    params['query'] = query_info['query'].format(namespace=namespace)
    response = requests.get(base_url, params=params)

    print(
        f"Código de estado para la consulta '{query_info['resource']}' en '{namespace}': {response.status_code}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = result_querys + generate_filename(
        namespace, query_info['resource'], timestamp)

    if response.headers.get('Content-Type', '').startswith('application/json'):
        data = response.json()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Los datos se han guardado en '{output_file}'")
    else:
        print(
            f"La respuesta para '{query_info['resource']}' en '{namespace}' no es JSON. Guardando respuesta de texto.")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"La respuesta se ha guardado en '{output_file}'")

# -------- SCRIPT BEGINING --------


# Configurar el parser de argumentos
parser = argparse.ArgumentParser(
    description='Listar los nombres de los pods y contenedores en un namespace específico.')
parser.add_argument('namespace', type=str, help='El nombre del namespace')

# Parsea los argumentos
args = parser.parse_args()

# Ejecuta el comando kubectl y captura la salida JSON
result = subprocess.run(['kubectl', 'get', 'pods', '-n',
                        args.namespace, '-o', 'json'], capture_output=True, text=True)

# Verifica si el comando fue exitoso
if result.returncode == 0:
    # Carga el JSON en un diccionario de Python
    pods_json = json.loads(result.stdout)

    # Crea un diccionario para almacenar los nombres de los pods y sus contenedores
    pods_containers = {}
    for pod in pods_json['items']:
        pod_name = pod['metadata']['name']
        containers = [container['name']
                      for container in pod['spec']['containers']]
        pods_containers[pod_name] = containers

    # Crea un diccionario con la clave "pods" y la lista de nombres de pods y contenedores
    pods_dict = {"pods": pods_containers}

    # Imprime el diccionario en formato JSON
    print(f"Pods: {json.dumps(pods_dict, indent=4)}")
else:
    print(f"Error al ejecutar kubectl: {result.stderr}")

# Definición de la estructura
namespaces = {
    args.namespace: {
        # Reemplaza con los nombres reales de los pods
        "pods": pods_dict["pods"],
        "queries": [
            {
                "resource": "memory-usage",
                "query": 'sum(container_memory_usage_bytes{{namespace="{namespace}", container!~".*wait-.*"}}) by (pod, container)',
                "unit": "bytes"
            },
            {
                "resource": "cpu-usage",
                "query": 'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", container!~"wait-.*"}}[1m])) by (pod, container)',
                "unit": "cores"
            },
            {
                "resource": "cpu-limit",
                "query": 'avg(kube_pod_container_resource_limits{{namespace="{namespace}", container!~"wait-.*", unit="core"}}) by (pod, container)',
                "unit": "cores"
            }
        ]
    }
    # Puedes agregar más namespaces aquí si es necesario
}

# Solicitar al usuario la duración
duration_input = input(
    "Ingrese la duración del rango (ej. '1h' para una hora, '30m' para 30 minutos): ")

try:
    duration_seconds = parse_duration(duration_input)
    start_epoch, end_epoch = get_epoch_range(duration_seconds)
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Definimos la URL base y los parámetros comunes
base_url = "http://localhost:30090/api/v1/query_range"
params = {
    "start": str(start_epoch),
    "end": str(end_epoch),
    "step": "14"
}

# Realizamos las consultas y guardamos los resultados
for namespace, info in namespaces.items():
    print(f"\nProcesando namespace: {namespace}")
    print(f"Pods en este namespace: {', '.join(info['pods'].keys())}")
    for query_info in info['queries']:
        fetch_and_save_query(base_url, params, query_info, namespace)
        print(
            f"  Recurso: {query_info['resource']}, Unidad: {query_info['unit']}")

print(
    f"\nRango de tiempo utilizado: {datetime.fromtimestamp(start_epoch, timezone.utc)} a {datetime.fromtimestamp(end_epoch, timezone.utc)}")