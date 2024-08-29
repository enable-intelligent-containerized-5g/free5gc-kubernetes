# GRAPH

## Structure

### Files

<details>
<summary>data</summary>

- **kubectl-data.json**: pod and services information.
- **tcpdump-data.log**: kubernetes trafict.
</details>

<details>
<summary>data-filtered</summary>

- **kubectl-result.json**: pod and services information filtered.
- **tcpdump-result.log**: kubernetes trafic filtered.
</details>

<details>
<summary>result</summary>

- **free5gc-graph.gexf**: graph in gexf format.
- **free5gc-graph.png**: graph in png format.
</details>

### Scripts

- **get-kubectl-data.py**: filter pod and services information.
- **get-tcpdump-data.py**: filter kubernetes trafic.
- **build-graph.py**: get the graph

## Steps

1. Run `sudo tcpdump -i <interface-name> > path/to/tcpdump-data.log` to get the kubernetes trafic.

2. Deploy the [free5gc-kubernetes proyect](../README.md).

3. Run `kubectl get po,svc -n <namespace> -o json > path/to/kubectl-data.json` to get de pod and services information.

4. Filter the kubectl-data.json running `python3 get-kubectl-data.py`.

5. Filter the tcpdump-data.json running `python3 get-tcpdump-data.py`.

6. Get the graph running `python3 build-graph.py`.