data "google_project" "project" {
}

resource "google_container_cluster" "kubeflow_cluster" {
  name = var.name
  location = var.zone
  description = "kubernetes test cluster"
  remove_default_node_pool = true
  initial_node_count = "1"

  maintenance_policy {
    daily_maintenance_window {
      start_time = "02:00"
    }
  }

  network_policy {
    enabled = false
  }

  addons_config {
    http_load_balancing {
      disabled = false
    }
  }
  cluster_autoscaling {
    enabled = false
  }
  resource_labels = {"mesh_id": "proj-${data.google_project.project.number}"}
}


resource "google_container_node_pool" "kubeflow_control_pool" {
  name       = "control-pool"
  location   = var.zone
  cluster    = var.name
  node_count = 1

  node_config {
    preemptible  = false
    machine_type = "e2-standard-4"

    oauth_scopes    = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}
