# =============================================================================
# Nexus Makefile
# Usage: make <target>
# =============================================================================

.PHONY: help dev-up dev-down simulate features train serve test lint clean \
        phase1 phase2 phase3 phase4 phase5 reset-db logs

SHELL        := /bin/bash
N            ?= 1000000       # default simulation events
MODEL        ?= recommender   # default retrain target
COMPOSE      := docker compose -f docker-compose.yaml
PYTHON       := python3

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
RESET  := \033[0m

help: ## Show this help
	@echo ""
	@echo "$(CYAN)Nexus — Marketplace Intelligence Platform$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Infrastructure ────────────────────────────────────────────────────────────
dev-up: ## Start full local stack (Kafka, Redis, PostgreSQL, MLflow, Grafana)
	@echo "$(CYAN)Starting Nexus local stack...$(RESET)"
	$(COMPOSE) up -d
	@echo "$(GREEN)Stack running. Services:$(RESET)"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  Grafana:   http://localhost:3000  (admin/nexus)"
	@echo "  Kafka UI:  http://localhost:8080"
	@echo "  Dagster:   http://localhost:3001"
	@echo "  Redis:     localhost:6379"
	@echo "  PostgreSQL: localhost:5432"

dev-down: ## Stop all local services
	$(COMPOSE) down

logs: ## Tail all service logs
	$(COMPOSE) logs -f

reset-db: ## Drop and recreate all databases (WARNING: deletes data)
	@read -p "This deletes all data. Are you sure? [y/N] " confirm && \
	[[ $$confirm == [yY] ]] && $(COMPOSE) down -v && $(COMPOSE) up -d

# ── Data Pipeline ─────────────────────────────────────────────────────────────
simulate: ## Generate N synthetic marketplace events (default N=1000000)
	@echo "$(CYAN)Generating $(N) events...$(RESET)"
	$(PYTHON) -m services.data_simulator.main generate --n-events $(N)

features: ## Materialise all feature views (batch + stream backfill)
	@echo "$(CYAN)Running feature materialisation...$(RESET)"
	$(PYTHON) -m services.feature_store.pipeline.main materialise

validate-data: ## Run Great Expectations data quality checks
	$(PYTHON) -m services.feature_store.pipeline.validate

# ── Training ──────────────────────────────────────────────────────────────────
train: ## Train all models (recommender, search, forecasting, fraud)
	@echo "$(CYAN)Training all models...$(RESET)"
	$(MAKE) train-recommender
	$(MAKE) train-search
	$(MAKE) train-forecasting
	$(MAKE) train-fraud

train-recommender: ## Train two-tower + ranker
	$(PYTHON) -m services.recommender.training.train --config configs/recommender.yaml

train-search: ## Train LTR model + build search index
	$(PYTHON) -m services.search.ltr.train --config configs/search.yaml

train-forecasting: ## Train N-BEATS + TFT forecasting models
	$(PYTHON) -m services.forecasting.pipeline.train --config configs/forecasting.yaml

train-fraud: ## Train GraphSAGE fraud detection model
	$(PYTHON) -m services.fraud_detection.models.train --config configs/fraud.yaml

retrain: ## Retrain a specific model (MODEL=recommender|search|forecasting|fraud)
	@echo "$(CYAN)Retraining $(MODEL)...$(RESET)"
	$(PYTHON) -m services.$(MODEL).training.train --incremental

# ── Serving ───────────────────────────────────────────────────────────────────
serve: ## Start the inference gateway (port 8000)
	@echo "$(CYAN)Starting Nexus inference gateway on :8000$(RESET)"
	uvicorn services.serving.gateway.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod: ## Start gateway in production mode
	uvicorn services.serving.gateway.main:app --host 0.0.0.0 --port 8000 \
		--workers 4 --loop uvloop --http h11

# ── Phase shortcuts ───────────────────────────────────────────────────────────
phase1: dev-up simulate features validate-data ## Phase 1: Data + Feature Store
	@echo "$(GREEN)Phase 1 complete!$(RESET)"

phase2: train ## Phase 2: Model Training
	@echo "$(GREEN)Phase 2 complete!$(RESET)"

phase3: serve ## Phase 3: Serving
	@echo "$(GREEN)Phase 3 running at :8000$(RESET)"

phase4: ## Phase 4: Experimentation (starts experiment dashboard)
	dagster dev -m services.experimentation

phase5: ## Phase 5: Monitoring (opens Grafana)
	@echo "Grafana: http://localhost:3000"
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || true

# ── Quality ───────────────────────────────────────────────────────────────────
test: ## Run full test suite
	pytest tests/ services/ -v --cov

test-unit: ## Unit tests only (fast)
	pytest tests/unit/ -v -x

test-integration: ## Integration tests (requires dev stack)
	pytest tests/integration/ -v

lint: ## Lint + format check
	ruff check . && ruff format --check .

format: ## Auto-format code
	ruff format . && ruff check --fix .

typecheck: ## Run mypy type checking
	mypy services/ shared/ --ignore-missing-imports

# ── Utilities ─────────────────────────────────────────────────────────────────
install: ## Install project in dev mode
	pip install -e ".[dev,graph,forecasting]"

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/

setup: ## First-time project setup
	bash setup.sh
	cp .env.example .env
	pip install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)Setup complete. Run 'make dev-up' to start the stack.$(RESET)"
