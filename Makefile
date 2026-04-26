.DEFAULT_GOAL := help
PYTHON        := python3
PIP           := pip3
PYTEST        := pytest
CONFIG_DIR    := config

# ─────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Agent Evaluator Crew — available commands"
	@echo "  ─────────────────────────────────────────"
	@echo "  make install           Install all Python dependencies"
	@echo "  make setup             Create config/ and reports/ directories"
	@echo "  make run               Run simple (Ollama) evaluation"
	@echo "  make run-crew          Run full multi-agent CrewAI evaluation"
	@echo "  make dashboard         Launch Streamlit dashboard on :8501"
	@echo "  make test              Run unit tests with coverage"
	@echo "  make integration-test  Run integration tests (needs OPENAI_API_KEY)"
	@echo "  make lint              Lint with ruff"
	@echo "  make format            Auto-format with ruff"
	@echo "  make security          Security scan with bandit"
	@echo "  make ci                Full local CI (lint + setup + test + security)"
	@echo "  make clean             Remove generated files and caches"
	@echo ""

# ─────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────
.PHONY: install
install:
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed."

.PHONY: setup
setup:
	mkdir -p $(CONFIG_DIR) reports tests
	@if [ ! -f $(CONFIG_DIR)/agents.yaml ] && [ -f agents.yaml ]; then \
		cp agents.yaml $(CONFIG_DIR)/agents.yaml; \
		echo "✅ Copied agents.yaml → $(CONFIG_DIR)/"; \
	fi
	@if [ ! -f $(CONFIG_DIR)/tasks.yaml ] && [ -f tasks.yaml ]; then \
		cp tasks.yaml $(CONFIG_DIR)/tasks.yaml; \
		echo "✅ Copied tasks.yaml  → $(CONFIG_DIR)/"; \
	fi
	@if [ ! -f tests/__init__.py ]; then touch tests/__init__.py; fi
	@if [ ! -f .env ] && [ -f .env.example ]; then \
		cp .env.example .env; \
		echo "⚠️  Created .env from .env.example — fill in your API keys."; \
	fi
	@echo "✅ Project structure ready."

# ─────────────────────────────────────────────────────────────────
# Run evaluations
# ─────────────────────────────────────────────────────────────────
.PHONY: run
run: setup
	$(PYTHON) eval_simple.py

.PHONY: run-crew
run-crew: setup
	$(PYTHON) eval_crew.py

# ─────────────────────────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────────────────────────
.PHONY: dashboard
dashboard:
	streamlit run dashboard.py --server.port 8501

# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────
.PHONY: test
test: setup
	$(PYTEST) tests/test_tools.py -v \
		--tb=short \
		--cov=tools \
		--cov-report=term-missing \
		--cov-report=xml:reports/coverage.xml \
		--junitxml=reports/unit-test-results.xml

.PHONY: integration-test
integration-test: setup
	$(PYTEST) tests/test_integration.py -v \
		-m integration \
		--tb=short \
		--junitxml=reports/integration-test-results.xml

.PHONY: test-all
test-all: test integration-test

# ─────────────────────────────────────────────────────────────────
# Lint / Format / Security
# ─────────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	ruff check .

.PHONY: format
format:
    ruff check --fix .
    ruff format .

.PHONY: security
security:
	mkdir -p reports
	bandit -r . --exclude ./.git,./tests,./reports -ll -f json \
		-o reports/bandit-report.json || true
	@echo "✅ Security report saved to reports/bandit-report.json"

# ─────────────────────────────────────────────────────────────────
# Full local CI
# ─────────────────────────────────────────────────────────────────
.PHONY: ci
ci: lint setup test security
	@echo ""
	@echo "✅ Full local CI passed."

# ─────────────────────────────────────────────────────────────────
# Clean
# ─────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache reports/ \
		evaluation_results.json evaluation_history.json \
		$(shell find . -name "*.pyc" -o -name "*.pyo")
	@echo "✅ Clean complete."