# Files
SCRIPT := coffee_analysis.py
REQ    := requirements.txt
TESTS  := test_coffee_analysis.py

.PHONY: install run test clean all

# Install dependencies into the current Python environment (required if do not use Dev Containers)
install:
	pip install --upgrade pip && pip install -r $(REQ)
	pip install black flake8 pytest pytest-cov

# Run the analysis script (saves plot PNGs)
run:
	python $(SCRIPT)

format:
	black $(SCRIPT) $(TESTS)

lint:
	flake8 --ignore=E203,E266,E501,W503 $(SCRIPT)

# Run tests
test:
	pytest -q $(TESTS)

# Remove caches and generated plot files
clean:
	rm -rf __pycache__ .pytest_cache .coverage
	rm -f by_coffee.png hour.png daily.png correlation.png cluster_pca_scatter_simple.png

# Do everything: install deps and run tests
all: install format lint test