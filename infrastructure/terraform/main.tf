# =============================================================================
# Nexus Platform - AWS EKS Cluster Provisioning
# Provisions the underlying Kubernetes cluster for the distributed ML platform.
# =============================================================================

provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "nexus-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = {
    Environment = "production"
    Project     = "nexus-ml-platform"
  }
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.28"

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  cluster_endpoint_public_access = true

  eks_managed_node_groups = {
    # General compute for FastAPI Serving Gateway & Dagster
    general_compute = {
      min_size     = 2
      max_size     = 5
      desired_size = 3
      instance_types = ["t3.xlarge"]
    }
    
    # High memory for Redis Online Store & Flink Stream Processing
    memory_optimized = {
      min_size     = 1
      max_size     = 3
      desired_size = 2
      instance_types = ["r5.2xlarge"]
      labels = {
        workload_type = "data-intensive"
      }
    }

    # GPU nodes for PyTorch/Ray Distributed Training & Triton Inference
    gpu_compute = {
      min_size     = 0
      max_size     = 2
      desired_size = 0
      instance_types = ["g4dn.xlarge"]
      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]
    }
  }

  tags = {
    Environment = "production"
    Project     = "nexus-ml-platform"
  }
}