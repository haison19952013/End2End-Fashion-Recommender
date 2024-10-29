In this tutorial, you will manage your HPP app by Helm.

## How-to Guide
```shell
kubectl create ns model-serving
kubectl create clusterrolebinding model-serving-admin-binding \
  --clusterrole=admin \
  --serviceaccount=model-serving:default \
  --namespace=model-serving

kubectl create clusterrolebinding anonymous-admin-binding \
  --clusterrole=admin \
  --user=system:anonymous \
  --namespace=model-serving
```

```shell
cd ocr_chart
helm upgrade --install hpp .
```
