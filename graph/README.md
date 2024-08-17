# GRAPH

## Structure

### Files

- **kubectldata.json**: pod and services information.
- **tcpdumpdata.log**: kubernetes trafict.
- **kubectlresult.json**: pod and services information filtered.
- **tcpdumpresult.log**: kubernetes trafic filtered.
- **free5gcgraph.gexf**: graph in gexf format.
- **free5gcgraph.png**: graph in png format.

### Scripts

- **getkubectldata.py**: filter pod and services information.
- **gettcpdumpdata.py**: filter kubernetes trafic.
- **buildfree5gcgraph.py**: get the graph

## Steps

1. Run `sudo tcpdump -i <interface-name> > path/to/tcpdumpdata.log` to get the kubernetes trafic.

2. Deploy the [free5gc-kubernetes proyect](../README.md).

3. Run `kubectl get po,svc -n <namespace> -o json > path/to/kubectldata.json` to get de pod and services information.

4. Filter the kubectldata.json running `python3 getkubectldata.py`.

5. Filter the tcpdumpdata.json running `python3 gettcpdumpdata.py`.

6. get the graph running `python3 buildfree5gcgraph.py`.