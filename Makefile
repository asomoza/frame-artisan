.PHONY: lint lint-fix format-check format check test test-gpu-light test-gpu-heavy test-all

# Run ruff linter
lint:
	uv run --extra dev ruff check src/ tests/

# Auto-fix linter issues
lint-fix:
	uv run --extra dev ruff check --fix src/ tests/

# Check formatting (no changes)
format-check:
	uv run --extra dev ruff format --check src/ tests/

# Auto-format code
format:
	uv run --extra dev ruff format src/ tests/

# Lint + format check (CI-friendly)
check: lint format-check

# Normal tests (no GPU required)
test:
	uv run --extra test pytest tests/ -v

# Light GPU tests (downloads OzzyGT/tiny_LTX2 ~67 MB on first run)
test-gpu-light:
	GPU_LIGHT=1 uv run --extra test pytest tests/ -v -m gpu_light -s

# Heavy GPU tests (requires real models in the app database)
test-gpu-heavy:
	GPU_HEAVY=1 uv run --extra test pytest tests/ -v -m gpu_heavy -s

# All tests (normal + light GPU + heavy GPU)
test-all:
	GPU_LIGHT=1 GPU_HEAVY=1 uv run --extra test pytest tests/ -v -s
