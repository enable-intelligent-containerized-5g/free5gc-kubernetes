import os
import json
import pickle
from Pod import Pod
from Interface import Interface

# Array of resources
resources = []

# Path to the results file
path_kubectl_results = os.path.join('kubectlresults.pkl')

# Path to the kubectldata
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

def read_kubectl_data():
    # Leer el archivo JSON
    with open(path_kubectl_data) as file:
        data = json.load(file)

        for item in data.get('items', []):
            # Get kind
            kind = item.get('kind', '')

            if kind == 'Pod':
                # Get metadata
                metadata = item.get('metadata', {})

                # Get name
                name = metadata.get('labels', {}).get('name', '')
                if not name:
                    name = metadata.get('labels', {}).get('nf', '')

                # Get interfaces
                annotations = metadata.get('annotations', {})
                network_status  = json.loads(annotations.get('k8s.v1.cni.cncf.io/network-status', ''))
                interfaces = []
                for net_s in network_status:
                    interface = net_s.get('interface', '')
                    ip_address = net_s.get('ips', [])[0]
                    interfaces.append(Interface(interface, ip_address))

                resources.append(Pod(kind, name, interfaces))

            elif kind == 'Service':
                service = None
            else:
                print('Kind unknow')

        # Save the results
        save_kubectl_results()

        # Print the results
        upload_kubectl_results()

if __name__ == "__main__":
    read_kubectl_data()