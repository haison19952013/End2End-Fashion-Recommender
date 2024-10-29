### Install Terraform for Ubuntu 22.04
https://computingforgeeks.com/how-to-install-terraform-on-ubuntu/

## How-to Guide
Authenticate with GCP
```shell
gcloud auth login
gcloud auth application-default login
```

List all GCP projects:
```shell
gcloud projects list
```

Set a specified project
```shell
gcloud config set project <project-id>
```

Check current project
```shell
gcloud config get-value project
```

## Provision a new VM
```shell
terraform init # to initialize the project
terraform plan # to see what will be created
terraform apply # to create the cluster
terraform destroy # to delete the cluster
```