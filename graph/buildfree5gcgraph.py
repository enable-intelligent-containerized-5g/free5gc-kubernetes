import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from Pod import Pod
from Interface import Interface

# Create empty graph
G = nx.Graph()

# Path to the kubectldata
path_kubectl_results = os.path.join('kubectlresults.pkl')
# Path to the kubectldata
path_tcpdump_results = os.path.join('tcpdumpresults.pkl')

# Open results
def get_data_pkl(path):
    # Deserialize the array of models with pickle
    with open(path, 'rb') as file:
        return pickle.load(file)

def create_free5gc_graph():
    # Get the pods
    pods = get_data_pkl(path_kubectl_results)
    connections = get_data_pkl(path_tcpdump_results)

    # Add the nodes
    for pod in pods:
        G.add_node(pod.name, label=pod.name)

    # Add the edge
    for conn in connections:
        src_pod = None
        dst_pod = None
        for pod in pods:
            for interface in pod.list_interfaces:
                if interface.ip_address == conn.src_ip:
                    src_pod = pod
                if interface.ip_address == conn.dst_ip:
                    dst_pod = pod
        if src_pod and dst_pod:
            G.add_edge(src_pod.name, dst_pod.name)

    nx.draw(G, with_labels=True)
    plt.savefig('free5gcgraph.png')
    nx.write_gexf(G, 'free5gcgraph.gexf')
    plt.show()

if __name__ == "__main__":
    create_free5gc_graph()
