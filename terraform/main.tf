terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "6.42.0"
    }
  }
}

provider "google" {
  region = "us-central1"
}

resource "google_project" "proj" {
  name            = "AI Test"
  project_id      = "ai-expr-alpha"
  billing_account = var.billing_account
}

resource "google_storage_bucket" "bucket" {
  name                        = "code_for_index"
  location                    = "EU"
  project                     = google_project.proj.project_id
  uniform_bucket_level_access = true
  hierarchical_namespace {
    enabled = true
  }
}

resource "google_storage_folder" "folder" {
  bucket = google_storage_bucket.bucket.name
  name   = "repo1/"
}

variable "billing_account" {
  type        = string
  description = "0XXX0-0XXX0-0XXX0, Replace with your billing account ID"
}
