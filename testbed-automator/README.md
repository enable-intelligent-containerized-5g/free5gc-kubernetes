# testbed-automator

## Summary

`install.sh`: automates the deployment of a single-node k8s cluster, configures cluster, installs various CNIs, configures ovs bridges and sets everything up for deployment of 5G core.

`uninstall.sh`: reverses install.sh

`kubeadm-config.yaml`: is use for the `install.sh` script to create the K8s cluster.

`multus-daemonset-thick.yaml`: Multus CNI config.

`kube-flannel.yaml`: Flannel CNI config.

## Deploying

```sh
# Create the cluster
bash install.sh

# Delete the cluster
bash uninstall.sh
```

## troubleshooting

## kudeadm init hangs

Check kubelet status
```sh
sudo systemctl status --no-pager --full kubelet.service
```

If you get error like the following:

> OCI runtime create failed: expected cgroupsPath to be of format \"slice:prefix:name\" for systemd cgroups, got \"/kubepods/burstable/..."

Then remove the containerd configuration and restart containerd.

```
sudo rm /etc/containerd/config.toml
systemctl restart containerd
```
