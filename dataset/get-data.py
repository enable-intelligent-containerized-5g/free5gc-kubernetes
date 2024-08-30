import re
import json
import argparse
import requests
import subprocess
from datetime import datetime, timezone, timedelta

data_path = "data/"

"""Time"""
timestamp = end_time = datetime.now()

def is_valid_duration(duration):
    # Regular expression pattern to match numbers followed by 'h' or 'm'
    pattern = r'^\d+[hm]$'
    return bool(re.match(pattern, duration))

def parse_duration(duration_str):
    """Convert a duration string (e.g. '1h', '30m') to seconds."""
    if is_valid_duration(duration_str):
        unit = duration_str[-1].lower()
        value = int(duration_str[:-1])
        if unit == 'h':
            return value * 3600
        elif unit == 'm':
            return value * 60
        else:
            raise ValueError(
                f"Invalid duration format: '{duration_str}'. Use 'h' for hours or 'm' for minutes.")
    else:
        raise ValueError(
                f"Invalid duration format: '{duration_str}'. Use 'h' for hours or 'm' for minutes.")


def get_epoch_range(duration, end_time):
    """Calculate the time range in epoch based on the duration"""
    # end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(seconds=duration)
    return int(start_time.timestamp()), int(end_time.timestamp())


def generate_filename(namespace, resource, duration, timestamp):
    """To generate a filename based on a namespace and a resource"""
    return f"{namespace}_{resource}_{duration}_{timestamp}.json"


def fetch_and_save_query(base_url, params, query_info, namespace, duration, timestamp):
    """Get and save the results in a JSON file"""
    print(f"\nResource: '{query_info['resource']}', Unit: '{query_info['unit']}', Namespace '{namespace}'")
    print(f"  Range ({duration}): from '{datetime.fromtimestamp(int(params['start'])).strftime('%Y-%m-%d %H:%M:%S')}'",
        f"to '{datetime.fromtimestamp(int(params['end'])).strftime('%Y-%m-%d %H:%M:%S')}'")
    params['query'] = query_info['query'].format(namespace=namespace)
    response = requests.get(base_url, params=params)

    print(
        f"  Status code: {response.status_code}")

    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = data_path + generate_filename(
        namespace, query_info['resource'], duration, timestamp.strftime("%Y-%m-%d_%H-%M-%S"))

    if response.headers.get('Content-Type', '').startswith('application/json'):
        data = response.json()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"    Data saved in '{output_file}'")
    else:
        print(
            f"    Response for '{query_info['resource']}' in '{namespace}' isn't JSON. Saving text response.")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"    Response save in '{output_file}'")

def main():
    """Set the parser"""
    default_namespace = "free5gc"
    default_duration = "1h"
    parser = argparse.ArgumentParser(
        description='Get the pods and containers of a namespace.')
    parser.add_argument('-n', '--namespace', dest='namespace', type=str, required=False, 
                        help=f'Namespace name (default is "{default_namespace}").', 
                        default=default_namespace)
    parser.add_argument('-t', '--time', dest='time', type=str, required=False, 
                        help=f"Duration in minutes or hours (e.g., '1h' for one hour, '30m' for 30 minutes). Default is '{default_duration}').", 
                        default=default_duration)

    """Parse the args"""
    args = parser.parse_args()
    arg_namespace = args.namespace
    arg_duration = args.time

    """Run the kubectl comand and get the output in a JSON format"""
    result = subprocess.run(['kubectl', 'get', 'pods', '-n',
                            arg_namespace, '-o', 'json'], capture_output=True, text=True)

    """Verify if the command was successful."""
    if result.returncode == 0:
        """Load the JSON in a dictionay of Python"""
        pods_json = json.loads(result.stdout)

        """Create a dictionary to save the pod and container names."""
        pods_containers = {}
        for pod in pods_json['items']:
            pod_name = pod['metadata']['name']
            containers = [container['name']
                        for container in pod['spec']['containers']]
            pods_containers[pod_name] = containers

        """Create a dictionary and store the list of pod and container names under the key pods"""
        pods_dict = {"pods": pods_containers}

        """Save pod list in JSON format"""
        pod_list = json.dumps(pods_dict, indent=4)
        with open(data_path + "pod-data.json", "w") as file:
            file.write(pod_list)
        print(f"Pods: {pod_list}")
    else:
        print(f"Error running kubectl comand: {result.stderr}")

    """Structure queries"""
    namespaces = {
        arg_namespace: {
            "pods": pods_dict["pods"],
            "queries": [
                {
                    "resource": "cpu-request",
                    "query": 'avg(kube_pod_container_resource_requests{{namespace="{namespace}", container!~"wait-.*", unit="core"}}) by (pod, container)',
                    "unit": "core"
                },
                {
                    "resource": "cpu-limit",
                    "query": 'avg(kube_pod_container_resource_limits{{namespace="{namespace}", container!~"wait-.*", unit="core"}}) by (pod, container)',
                    "unit": "core"
                },
                {
                    "resource": "memory-limit",
                    "query": 'sum(avg(kube_pod_container_resource_limits{{namespace="{namespace}", container!~"wait-.*", unit="bytes"}})) by (pod, container)',
                    "unit": "bytes"
                },
                {
                    "resource": "cpu-usage",
                    "query": 'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}", container!~"wait-.*"}}[1m])) by (pod, container)',
                    "unit": "core"
                },
                {
                    "resource": "memory-request",
                    "query": 'avg(kube_pod_container_resource_requests{{namespace="{namespace}", container!~".*wait-.*", unit="byte"}}) by (pod, container)',
                    "unit": "byte"
                },
                {
                    "resource": "memory-limit",
                    "query": 'avg(kube_pod_container_resource_limits{{namespace="{namespace}", container!~".*wait-.*", unit="byte"}}) by (pod, container)',
                    "unit": "byte"
                },
                {
                    "resource": "memory-usage",
                    "query": 'sum(container_memory_usage_bytes{{namespace="{namespace}", container!~".*wait-.*"}}) by (pod, container)',
                    "unit": "byte"
                },
                {
                    "resource": "receive-packets",
                    "query": 'sum(rate(container_network_receive_packets_total{{namespace="{namespace}", container!~".*wait-.*"}}[1m])) by (pod, container)',
                    "unit": "packets/s"
                },
                {
                    "resource": "transmit-packets",
                    "query": 'sum(rate(container_network_transmit_packets_total{{namespace="{namespace}", container!~".*wait-.*"}}[1m])) by (pod, container)',
                    "unit": "packets/s"
                },
                {
                    "resource": "total-packets",
                    "query": 'sum(rate(container_network_receive_packets_total{{namespace="{namespace}", container!~".*wait-.*"}}[1m])) by (pod, container) + sum(rate(container_network_transmit_packets_total{{namespace="{namespace}", container!~".*wait-.*"}}[1m])) by (pod, container)',
                    "unit": "packets/s"
                },
                {
                    "resource": "receive-packets-dropped",
                    "query": 'sum(rate(container_network_receive_packets_dropped_total{{namespace="{namespace}", container!~".*wait-.*"}}[1m])) by (pod, container)',
                    "unit": "packets/s"
                },
                {
                    "resource": "transmit-packets-dropped",
                    "query": 'sum(rate(container_network_transmit_packets_dropped_total{{namespace="{namespace}", container!~".*wait-.*"}}[1m])) by (pod, container)',
                    "unit": "packets/s"
                },
                {
                    "resource": "total-packets-dropped",
                    "query": 'sum(rate(container_network_receive_packets_dropped_total{{namespace="{namespace}", container!~".*wait-.*"}}[1m])) by (pod, container) + sum(rate(container_network_transmit_packets_dropped_total{{namespace="{namespace}", container!~".*wait-.*"}}[1m])) by (pod, container)',
                    "unit": "packets/s"
                }
            ]
        }
        # You can add more namespaces here if necessary.
    }

    try:
        duration_seconds = parse_duration(arg_duration)
        start_epoch, end_epoch = get_epoch_range(duration_seconds, timestamp)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    """URL base y and params"""
    base_url = "http://localhost:30090/api/v1/query_range"
    params = {
        "start": str(start_epoch),
        "end": str(end_epoch),
        "step": "14"
    }

    """Make the queries and save the results"""
    for namespace, info in namespaces.items():
        print(f"Processing namespace: {namespace}")
        print(f"Pods in this namespace:")
        for pod_name, pod_values in info['pods'].items():
            print(f"  {pod_name}: {', '.join(pod_values)}")

        for query_info in info['queries']:
            fetch_and_save_query(base_url, params, query_info, namespace, arg_duration, timestamp)

    print(
        f"\nTime range used '{arg_duration}': From '{datetime.fromtimestamp(start_epoch, timezone.utc)}' to '{datetime.fromtimestamp(end_epoch, timezone.utc)}'")
    
if __name__ == '__main__':
    main()
