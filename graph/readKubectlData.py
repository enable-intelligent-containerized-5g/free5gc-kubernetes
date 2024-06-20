import json
import networkx as nx
import matplotlib.pyplot as plt

# kubectl get po,svc -n test -o json > resources.json

# Crea un grafo vacío
G = nx.Graph()
# Abre el archivo JSON
with open('resources.json') as f:
    data = json.load(f)

# Agrega nodos para los pods
for item in data['items']:
    if item['kind'] == 'Pod':
        name = item.get('metadata', {}).get('labels', {}).get('name', '')
        if not name:
            name = item.get('metadata', {}).get('labels', {}).get('nf', '')

        G.add_node(name, label=item['kind'])

# # Agrega bordes para los servicios
# for item in data['items']:
#     if item['kind'] == 'Service':
#         for spec in item['spec']['ports']:
#             G.add_edge(item['metadata']['name'], spec['port'], label=item['metadata']['name'])

# Imprime el grafo
# print(nx.info(G))

nx.draw(G, with_labels=True)
plt.savefig('graph.png')
plt.show()
