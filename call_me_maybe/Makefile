VENV_DIR = .venv
VENV_BIN = $(VENV_DIR)/bin
PYTHON = $(VENV_BIN)/python3
UV = uv
SRC_DIR = src
MAIN = src/__main__.py

CACHE_ENV = UV_CACHE_DIR=/goinfre/eseragio/.cache/uv HF_HOME=/goinfre/eseragio/.cache/huggingface


all: install

$(VENV_DIR):
	@echo "Creating virtual environment"
	$(UV) venv $(VENV_DIR)


install: $(VENV_DIR)
	@echo "Install dependencies"
	@$(CACHE_ENV) uv sync
	@echo "everything installed"

run:
	@$(CACHE_ENV) uv run python -m src

debug:
	@$(CACHE_ENV) uv run python -m pdb -m src

lint:
	@$(CACHE_ENV) uv run flake8 $(SRC_DIR)
	@$(CACHE_ENV) uv run mypy $(SRC_DIR) --warn-return-any\
	--warn-unused-ignores \
	--ignore-missing-imports \
	--disallow-untyped-defs \
	--check-untyped-defs

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

fclean: clean
	rm -rf $(VENV_DIR)

.PHONY: all install run debug lint clean fclean
