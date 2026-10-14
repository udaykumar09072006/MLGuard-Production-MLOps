.PHONY: help install lint test test-unit test-api run-api run-dashboard docker-up docker-down demo

help:
	@echo "MLGuard Platform Commands:"
	@echo "  make install        Install all Python dependencies"
	@echo "  make lint           Run Ruff linter and code formatter check"
	@echo "  make test           Run full pytest suite"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-api       Run API integration tests"
	@echo "  make run-api        Start FastAPI backend service locally"
	@echo "  make run-dashboard  Start Streamlit dashboard locally"
	@echo "  make docker-up      Start all services with Docker Compose"
	@echo "  make docker-down    Stop all Docker Compose services"
	@echo "  make demo           Generate sample datasets and run demo training"

install:
	pip install -r requirements.txt

lint:
	ruff check .
	ruff format --check .

test:
	pytest tests/ -v --cov=app

test-unit:
	pytest tests/unit/ -v

test-api:
	pytest tests/api/ -v

run-api:
	uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:
	streamlit run app/dashboard/Home.py --server.port 8501

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

demo:
	python scripts/generate_sample_data.py
	python scripts/seed_demo.py
