# testbed-automator
`Vagrantfile`: create the ubuntu 20.04 VM.

`install.sh`: automates the deployment of a single-node k8s cluster, configures cluster, installs various CNIs, configures ovs bridges and sets everything up for deployment of 5G core.

`uninstall.sh`: reverses install.sh

`kubeadm-config.yaml`: is use for the `install.sh` script to create the K8s cluster.

# troubleshooting

## kudeadm init hangs

Check kubelet status
```bash
sudo systemctl status --no-pager --full kubelet.service
```

If you get error like the following:

> OCI runtime create failed: expected cgroupsPath to be of format \"slice:prefix:name\" for systemd cgroups, got \"/kubepods/burstable/..."

Then remove the containerd configuration and restart containerd.

```
sudo rm /etc/containerd/config.toml
systemctl restart containerd
```
