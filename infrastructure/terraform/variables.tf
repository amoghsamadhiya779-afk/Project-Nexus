variable "aws_region" {
  description = "The AWS region to deploy the Nexus EKS cluster."
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "Name of the EKS cluster."
  type        = string
  default     = "nexus-production-cluster"
}