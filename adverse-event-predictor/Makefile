# ============================================================
#  Adverse Event Risk Predictor — Makefile
#  Usage: make <target>
# ============================================================

PYTHON      := python3
PROJECT_DIR := $(shell pwd)
PYTHONPATH  := $(PROJECT_DIR)/python:$(PROJECT_DIR)/api

.PHONY: help setup data train evaluate api dashboard test lint docker-build docker-up clean all

# ── Default: show help ─────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "⚕  Adverse Event Risk Predictor — Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make setup        Install Python dependencies"
	@echo "  make data         Generate synthetic MIMIC-III data"
	@echo "  make train        Train all ML models"
	@echo "  make evaluate     Run evaluation + SHAP analysis"
	@echo "  make pipeline     Full pipeline: data → train → evaluate"
	@echo "  make api          Start FastAPI server (port 8000)"
	@echo "  make dashboard    Start Streamlit dashboard (port 8501)"
	@echo "  make test         Run pytest test suite"
	@echo "  make lint         Run ruff linter"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start all services via docker-compose"
	@echo "  make docker-down  Stop docker-compose services"
	@echo "  make clean        Remove generated files"
	@echo "  make all          Full pipeline + tests"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────
setup:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Setup complete"

setup-dev: setup
	pip install pytest pytest-cov ruff mypy httpx

# ── Data ───────────────────────────────────────────────────────────────────────
data:
	@echo "🗄  Generating synthetic MIMIC-III data..."
	mkdir -p data/processed data/raw models
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) python/data_loader.py \
		--mode synthetic \
		--output data/processed/ae_dataset.csv \
		--n-patients 5000
	@echo "✓ Data written to data/processed/ae_dataset.csv"

data-real:
	@echo "🗄  Loading real MIMIC-III data from PostgreSQL..."
	@echo "   Ensure MIMIC_DB_URL is set in .env"
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) python/data_loader.py \
		--mode real \
		--input data/processed/ae_features_clean.csv \
		--output data/processed/ae_dataset.csv

# ── Training ───────────────────────────────────────────────────────────────────
train:
	@echo "🤖 Training all ML models..."
	mkdir -p models
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) python/train_model.py \
		--source synthetic \
		--cv 5 \
		--version v1
	@echo "✓ Best model saved to models/ae_model.pkl"

train-real:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) python/train_model.py \
		--source csv \
		--cv 5 \
		--version v1

# ── Evaluation ─────────────────────────────────────────────────────────────────
evaluate:
	@echo "📊 Running model evaluation + SHAP..."
	mkdir -p data/processed/reports
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) python/evaluate_model.py
	@echo "✓ Reports saved to data/processed/reports/"

# ── Full Pipeline ──────────────────────────────────────────────────────────────
pipeline: data train evaluate
	@echo ""
	@echo "✅ Full pipeline complete!"
	@echo "   Model:   models/ae_model.pkl"
	@echo "   Reports: data/processed/reports/"

# ── API ────────────────────────────────────────────────────────────────────────
api:
	@echo "🚀 Starting FastAPI server at http://localhost:8000"
	@echo "   Docs: http://localhost:8000/docs"
	PYTHONPATH=$(PYTHONPATH) uvicorn api.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

api-prod:
	PYTHONPATH=$(PYTHONPATH) uvicorn api.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--workers 4 \
		--log-level warning

# ── Dashboard ──────────────────────────────────────────────────────────────────
dashboard:
	@echo "📈 Starting Streamlit dashboard at http://localhost:8501"
	PYTHONPATH=$(PYTHONPATH) streamlit run dashboard/streamlit_app.py \
		--server.port 8501 \
		--server.address 0.0.0.0 \
		--browser.gatherUsageStats false

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Running test suite..."
	PYTHONPATH=$(PYTHONPATH) pytest tests/ \
		-v \
		--tb=short \
		-x \
		--ignore=tests/test_pipeline.py  # skip API tests if model not trained

test-all:
	PYTHONPATH=$(PYTHONPATH) pytest tests/ \
		-v \
		--tb=short \
		--cov=python \
		--cov=api \
		--cov-report=term-missing \
		--cov-report=html:data/processed/coverage

test-fast:
	PYTHONPATH=$(PYTHONPATH) pytest tests/ \
		-v -x \
		-k "not TestAPI and not test_full_training_pipeline"

# ── Lint ───────────────────────────────────────────────────────────────────────
lint:
	@echo "🔍 Running linter..."
	ruff check python/ api/ dashboard/ tests/ --fix
	@echo "✓ Lint complete"

format:
	ruff format python/ api/ dashboard/ tests/

typecheck:
	mypy python/ api/ --ignore-missing-imports

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	@echo "🐳 Building Docker image..."
	docker build -t ae-predictor:latest --target runtime .

docker-up:
	@echo "🐳 Starting all services..."
	docker-compose up -d
	@echo "Services started:"
	@echo "  API:       http://localhost:8000"
	@echo "  Dashboard: http://localhost:8501"
	@echo "  MLflow:    http://localhost:5000"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-ps:
	docker-compose ps

# ── Notebook ───────────────────────────────────────────────────────────────────
notebook:
	@echo "📓 Generating Jupyter notebook..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) notebooks/generate_notebook.py
	jupyter notebook notebooks/exploratory_analysis.ipynb

# ── SQL Extraction ─────────────────────────────────────────────────────────────
sql-extract:
	@echo "🗄  Running SQL feature extraction on MIMIC-III..."
	@echo "   Requires PostgreSQL with MIMIC-III loaded"
	psql $$MIMIC_DB_URL -f data/raw/extract_mimic.sql
	psql $$MIMIC_DB_URL -c "\COPY ae_model_features TO 'data/raw/ae_model_features.csv' CSV HEADER"
	@echo "✓ Raw features exported to data/raw/ae_model_features.csv"

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf data/processed/*.csv
	rm -rf data/processed/reports/
	rm -rf models/*.pkl models/*.json
	rm -rf __pycache__ **/__pycache__ **/**/__pycache__
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "✓ Clean complete"

clean-docker:
	docker-compose down -v
	docker rmi ae-predictor:latest 2>/dev/null || true

# ── All ────────────────────────────────────────────────────────────────────────
all: setup pipeline test
	@echo ""
	@echo "🎉 All steps complete!"
