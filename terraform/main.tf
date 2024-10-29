# Ref: https://github.com/terraform-google-modules/terraform-google-kubernetes-engine/blob/master/examples/simple_autopilot_public
# To define that we will use GCP
terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.80.0" // Provider version
    }
  }
  required_version = "1.9.8" // Terraform version
}

// The library with methods for creating and
// managing the infrastructure in GCP, this will
// apply to all the resources in the project
provider "google" {
  project     = var.project_id
  region      = var.region
}

// Google Kubernetes Engine
# resource "google_container_cluster" "rec-sys-services" {
#   name     = "${var.project_id}-gke-rec-sys-services"
#   location = var.region
 
#   // Enabling Autopilot for this cluster
#   enable_autopilot = false
  
#   # // Enable Istio (beta)
#   # // https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/container_cluster#nested_istio_config
#   # // not yet supported on Autopilot mode
#   # addons_config {
#   #   istio_config {
#   #     disabled = false
#   #     auth     = "AUTH_NONE"
#   #   }
#   # }
# }

resource "google_compute_instance" "jenkins_service" {
  name         = var.instance_name
  machine_type = var.machine_type
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = var.boot_disk_image
      size = var.boot_disk_size
    }
  }
  # Important: need to define this one to allow external IP
  network_interface {
    network = "default"

    access_config {
      // Ephemeral public IP
    }
  }

  metadata = {
    ssh-keys = var.ssh_keys
  }
}

resource "google_compute_firewall" "firewall_jenkins_and_services" {
  name    = var.firewall_name
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8081", "50000", "16686", "8501"] // 8081, 50000: Jenkins, 16686: Jaeger, 8501: Streamlit
  }

  source_ranges = ["0.0.0.0/0"] // Allow all traffic
}