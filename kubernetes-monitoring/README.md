Before to install prometheus and grafana, you must install **kube-state-metrics**:

```sh
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm install kube-state-metrics prometheus-community/kube-state-metrics --namespace kube-system
```

Installation:

```sh
kubectl apply -k kubernetes-monitoring/
```