#!/usr/bin/env python3
"""
=============================================================================
Nexus Project Workspace Structure Generator
Creates every subdirectory and dynamically places __init__.py package placeholders.
Natively cross-platform (Windows, macOS, Linux).
=============================================================================
"""

import os
import sys
from pathlib import Path
import argparse

# ANSI escape codes for rich terminal feedback
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_status(message: str, color_code: str = "") -> None:
    """Prints a status message. Disables colors if running on unsupported terminals."""
    # Check if stdout is a TTY and we aren't on Windows without ANSICON (or use basic fallback)
    supports_color = sys.platform != "win32" or "ANSICON" in os.environ or os.environ.get("TERM") == "xterm" or os.environ.get("COLORTERM") is not None
    if supports_color and color_code:
        print(f"{color_code}{message}{Colors.END}")
    else:
        print(message)

# All target subdirectories for Nexus architecture
DIRS = [
    # Shared
    "shared/schemas",
    "shared/utils",
    "shared/monitoring",

    # Feature Store
    "services/feature_store/api",
    "services/feature_store/core",
    "services/feature_store/registry",
    "services/feature_store/pipeline",
    "services/feature_store/storage",
    "services/feature_store/tests",

    # Data Simulator
    "services/data_simulator/generators",
    "services/data_simulator/streams",
    "services/data_simulator/loaders",

    # Recommender
    "services/recommender/candidate_gen",
    "services/recommender/ranking",
    "services/recommender/embedding",
    "services/recommender/training",
    "services/recommender/tests",

    # Search
    "services/search/indexer",
    "services/search/retrieval",
    "services/search/ltr",
    "services/search/reranker",
    "services/search/tests",

    # Forecasting
    "services/forecasting/models",
    "services/forecasting/pipeline",
    "services/forecasting/causal",
    "services/forecasting/tests",

    # Fraud Detection
    "services/fraud_detection/graph",
    "services/fraud_detection/models",
    "services/fraud_detection/hitl",
    "services/fraud_detection/tests",

    # Experimentation
    "services/experimentation/ab_testing",
    "services/experimentation/metrics",
    "services/experimentation/guardrails",
    "services/experimentation/quasi",
    "services/experimentation/tests",

    # Serving Gateway
    "services/serving/gateway",
    "services/serving/feature_fetcher",
    "services/serving/model_server",
    "services/serving/cache",

    # SDK
    "sdk/nexus_client",

    # Infrastructure
    "infrastructure/docker/feature_store",
    "infrastructure/docker/serving",
    "infrastructure/docker/simulator",
    "infrastructure/kubernetes/feature-store",
    "infrastructure/kubernetes/serving",
    "infrastructure/kubernetes/monitoring",
    "infrastructure/kubernetes/kafka",
    "infrastructure/terraform",

    # Tests
    "tests/unit",
    "tests/integration",
    "tests/e2e",

    # CI/CD
    ".github/workflows",

    # Docs
    "docs/architecture",
    "docs/api"
]

def generate_workspace(root_path: Path, dry_run: bool = False) -> None:
    """Executes directory setup and places empty __init__.py files in module packages."""
    print_status(f"🛠️ Starting workspace setup in: {root_path.resolve()}", Colors.CYAN)
    if dry_run:
        print_status("⚠️ RUNNING IN DRY-RUN MODE: No changes will be written to disk.", Colors.YELLOW)
    
    dir_count = 0
    init_count = 0

    # Ensure root path exists
    if not dry_run:
        root_path.mkdir(parents=True, exist_ok=True)

    # 1. Process standard subdirectories
    for relative_dir in DIRS:
        target_dir = root_path / relative_dir
        
        # Create directory recursively (equivalent to mkdir -p)
        if not target_dir.exists():
            if not dry_run:
                target_dir.mkdir(parents=True, exist_ok=True)
            dir_count += 1
            if dry_run:
                print(f"[Dry Run] Would create folder: {relative_dir}")
        
        # Check if folder belongs to modular Python workspace packages
        is_package = any(relative_dir.startswith(p) for p in ["services/", "shared/", "sdk/"])
        if is_package:
            init_file = target_dir / "__init__.py"
            if not init_file.exists():
                if not dry_run:
                    init_file.touch(exist_ok=True)
                init_count += 1
                if dry_run:
                    print(f"[Dry Run] Would create python init module: {relative_dir}/__init__.py")

    # 2. Process top-level Python package workspace inits
    top_level_packages = ["shared", "sdk"]
    for package in top_level_packages:
        top_init_file = root_path / package / "__init__.py"
        # Only create if the base directory was declared
        if (root_path / package).exists() or dry_run:
            if not top_init_file.exists():
                if not dry_run:
                    top_init_file.touch(exist_ok=True)
                init_count += 1
                if dry_run:
                    print(f"[Dry Run] Would create top-level init module: {package}/__init__.py")

    # Complete execution log output
    print_status("\n✅ Directory structure processing complete.", Colors.GREEN)
    if not dry_run:
        print_status(f"  Created folders: {dir_count}", Colors.BOLD)
        print_status(f"  Created python package files: {init_count}", Colors.BOLD)
    else:
        print_status(f"  Dry run completed. Total proposed directories: {len(DIRS)}", Colors.CYAN)

    print_status("\nNext steps to configure Nexus on your environment:", Colors.YELLOW)
    print_status("  1. Copy Phase 1 files into their respective locations (see PLACEMENT.md)")
    
    # OS-Specific commands for convenience
    if sys.platform == "win32":
        print_status("  2. Run: copy .env.example .env (and adjust credentials accordingly)")
        print_status("  3. Run: docker-compose up -d")
    else:
        print_status("  2. Run: cp .env.example .env && nano .env")
        print_status("  3. Run: make dev-up")
        print_status("  4. Run: make simulate N=100000")
        print_status("  5. Run: make features")

def main():
    parser = argparse.ArgumentParser(
        description="Convert shell-based folder directories structure into native platform implementations."
    )
    parser.add_argument(
        "--root", "-r",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Specify alternative root path target directory. Defaults to script folder parent."
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Simulate the task and print outputs without generating directories or writing files."
    )
    
    args = parser.parse_args()
    target_root = Path(args.root)
    
    try:
        generate_workspace(target_root, dry_run=args.dry_run)
    except Exception as e:
        print_status(f"❌ Error encountered during structure deployment: {e}", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main()