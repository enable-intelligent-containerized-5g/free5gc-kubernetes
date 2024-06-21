import os
import json
import pickle
import ast

# Interfaz model
class Interface:
    def __init__(self, name, ip_address):
        self.name = name
        self.ip_address = ip_address

    def __str__(self):
        return f"{self.name}: {self.ip_address}"

# Resource model
class Resource:
    def __init__(self, kind, name, list_interfaces=None):
        self.kind = kind
        self.name = name
        if list_interfaces is None:
            list_interfaces = []
        self.list_interfaces = list_interfaces

    def __str__(self):
        # Convert the list of interfaces in a comma separated string
        list_interfaces_str = ', '.join(str(interface) for interface in self.list_interfaces)
        # Devolver una cadena que representa el objeto
        return f"Resource {self.name} type {self.kind} with interfaces [{list_interfaces_str}]"

# Array of resources
resources = []

# Path to the results file
path_kubectl_results = os.path.join('kubectlresults.pkl')

# Path to the tcpdumpdata
# Use: kubectl get po,svc -n <namespace> -o json > path/to/kubectldata.json
path_kubectl_data = os.path.join('kubectldata.json')

# Save the results
def save_kubectl_results():
    with open(path_kubectl_results, 'wb') as file:
        pickle.dump(resources, file)

# Read the results
def upload_kubectl_results():
    # Deserialize the array of models with pickle
    with open(path_kubectl_results, 'rb') as file:
        models = pickle.load(file)

    # Print the results
    print_results(models)

# Print the result
def print_results(models):
    for model in models:
        print(model)

# Test
# interfaces = [Interface('eth0', '192.168.0.1'), Interface('eth1', '10.0.0.1')]
# obj = Resource('kind1', 'name1', interfaces)
# print(obj)

def read_kubectl_data():
    # Leer el archivo JSON
    with open(path_kubectl_data) as file:
        data = json.load(file)

        for item in data.get('items', []):
            if item.get('kind', '') == 'Pod':
                # Get metadata
                metadata = item.get('metadata', {})

                # Get kind
                kind = item.get('kind', '')

                # Get name
                name = metadata.get('labels', {}).get('name', '')
                if not name:
                    name = metadata.get('labels', {}).get('nf', '')

                # Get interfaces
                annotations = metadata.get('annotations', {})
                network_status  = json.loads(annotations.get('k8s.v1.cni.cncf.io/network-status', ''))
                interfaces = []
                interfaces.append(Interface("eth0", "192.168.20.21"))
                # for net_s in network_status:
                #     name = interface['name']
                #     ip_address = interface['ip_address']
                #     interfaces.append(Interface(name, ip_address))

                resources.append(Resource(kind, name, interfaces))

                # ----------------------

                # Diccionario proporcionado
                # annotations = {
                #     "k8s.v1.cni.cncf.io/network-status": "[{\n    \"name\": \"cbr0\",\n    \"interface\": \"eth0\",\n    \"ips\": [\n        \"10.244.0.163\"\n    ],\n    \"mac\": \"d6:12:48:b7:13:9c\",\n    \"default\": true,\n    \"dns\": {},\n    \"gateway\": [\n        \"10.244.0.1\"\n    ]\n},{\n    \"name\": \"test/n2network\",\n    \"interface\": \"n3\",\n    \"ips\": [\n        \"10.10.2.1\"\n    ],\n    \"mac\": \"0a:58:0a:0a:02:01\",\n    \"dns\": {}\n}]",
                #     "k8s.v1.cni.cncf.io/networks": "[ { \"name\": \"n2network\", \"interface\": \"n3\", \"ips\": [ \"10.10.2.1/24\" ] } ]"
                # }

                # # Extraer y convertir el valor de "k8s.v1.cni.cncf.io/network-status" a un objeto Python
                # network_status = json.loads(annotations["k8s.v1.cni.cncf.io/network-status"])

                # # Extraer y convertir el valor de "k8s.v1.cni.cncf.io/networks" a un objeto Python
                # networks = json.loads(annotations["k8s.v1.cni.cncf.io/networks"])

                # # Ahora puedes acceder a los valores específicos
                # for status in network_status:
                #     print(f"Name: {status['name']}")
                #     print(f"Interface: {status['interface']}")
                #     print(f"IPs: {status['ips']}")
                #     print(f"MAC: {status['mac']}")
                #     print(f"Default: {status.get('default', False)}")
                #     print(f"DNS: {status['dns']}")
                #     print(f"Gateway: {status.get('gateway', [])}")
                #     print()  # Separador

                # for network in networks:
                #     print(f"Name: {network['name']}")
                #     print(f"Interface: {network['interface']}")
                #     print(f"IPs: {network['ips']}")
                #     print()  # Separador

        # Save the results
        save_kubectl_results()

        # Print the results
        upload_kubectl_results()

if __name__ == "__main__":
    read_kubectl_data()