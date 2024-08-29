import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt


# Open results
def get_data_pkl(path):
    # Deserialize the array of models with pickle
    with open(path, 'rb') as file:
        return pickle.load(file)

def main():
    # Create empty graph
    G = nx.Graph()

    # Result path
    result_path = "result/"
    data_path = "data-filtered/"

    # Path to the kubectldata
    path_kubectl_results = os.path.join(f"{data_path}kubectl-results.pkl")
    # Path to the kubectldata
    path_tcpdump_results = os.path.join(f"{data_path}tcpdump-results.pkl")

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
    plt.savefig(f"{result_path}free5gc-graph.png")
    nx.write_gexf(G, f"{result_path}free5gc-graph.gexf")
    plt.show()

if __name__ == "__main__":
    main()
