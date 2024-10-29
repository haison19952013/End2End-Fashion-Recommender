// Variables to use accross the project
// which can be accessed by var.project_id
variable "project_id" {
  description = "The project ID to host the cluster in"
  default     = "mle-course-437404" # remember to change this based on your project ID in GCP
}

variable "region" {
  description = "The region the cluster in"
  default     = "asia-southeast1"
}

# Config for the compute engine instance

variable "instance_name" {
  description = "Name of the instance"
  default     = "jenkins-and-services"
}

variable "machine_type" {
  description = "Machine type for the instance"
  default     = "e2-standard-2"
}

variable "zone" {
  description = "Zone for the instance"
  default     = "asia-southeast1-a"
}

variable "boot_disk_image" {
  description = "Boot disk image for the instance"
  default     = "ubuntu-os-cloud/ubuntu-2204-lts"
}

variable "boot_disk_size" {
  description = "Boot disk size for the instance"
  default     = 50
}

variable "firewall_name" {
  description = "Name of the firewall rule"
  default     = "jenkins-and-services-firewall" 
}

variable "ssh_keys" {
  description = "value of the ssh key"
  default = "haison19952013:ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINBtuJ1/JDbSXdKan+4fPgfrZpUoe0I+scgqHMA+DIyY haison19952013@gmail.com"
}