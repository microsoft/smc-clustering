# Copilot Instructions

## Build, Test, and Lint

This project uses **uv** as the package manager and **Python 3.13**.

```bash
# Install dependencies
uv sync --locked

# Run the full CI suite (format check → import sort → lint → tests)
./test.sh

# Individual commands
uv run ruff format --check    # Check formatting
uv run ruff format            # Auto-format
uv run ruff check --fix       # Lint with auto-fix
uv run pytest                 # Run all tests
uv run pytest tests/jsonlm/test_grammar.py          # Single test file
uv run pytest tests/jsonlm/test_grammar.py::test_masks_basic_states -v  # Single test
```

## Architecture

The package `smc_clustering` (under `src/`) implements entity linking by combining a grammar-constrained language model with Sequential Monte Carlo clustering.

### Three subpackages

- **`clustering/`** — JAX-based SMC (and MCMC/agglomerative) clustering. Maintains particles as sets of hashable `Cluster` frozensets with pluggable surrogate score models (Gaussian, Bernoulli, Bigram). The JSON-LM scorer integrates here via `score_entities_batched()`.
- **`jsonlm/`** — A PyTorch/Lightning transformer language model that scores entity dicts under a strict JSON grammar with `<K>` (key) and `<V>` (value) sentinel tokens.
- **`diffusion/`** — Flax/JAX variational diffusion with a `SetFormer` permutation-invariant architecture. Research exploration; not the primary scoring path.

### How they connect

The JSON-LM computes log-likelihoods for candidate entity clusters, which the SMC clusterer uses as its score function to perform incremental entity assignment with resampling.

### Code layout beyond `src/`

- **`experiments/`** — Synthetic experiment runners (Gaussian, circles) using JAX.
- **`scripts/`** — Evaluation pipelines that load benchmark configs (via MS-KeBAB), pre-trained JSON-LM checkpoints, and run SMC clustering with metrics.
- **`notebooks/`** — Jupyter notebooks for visualization and demos.
- **`tests/`** — Primarily tests for the `jsonlm` subpackage; integration tests do real training (no mocks).

## Key Conventions

### Framework split
JAX/Flax is used in `clustering/` and `diffusion/`; PyTorch/Lightning is used in `jsonlm/`. These do not share tensor objects—they solve different parts of the pipeline.

### Docstrings and type annotations
Ruff enforces Google-style docstrings (`D` rules) and type annotations (`ANN` rules) on all public functions. Prefix unused arguments with `_`.

### Formatting
Ruff handles formatting: double quotes, spaces, 105-char line length, 2 blank lines after imports (`isort` `lines-after-imports = 2`).

### Entity canonicalization
Entities (Python dicts) are always sorted by key, then by value, before serialization. This is critical for deterministic scoring—never skip this step.

### Grammar-constrained generation
Token masks from `GrammarAutomaton` + `allowed_token_mask()` must be applied before any softmax during both training and inference. Invalid tokens get `-inf` before `log_softmax`.

### Config pattern
Experiments use JSON benchmark configs loaded via the external `mskebab.Benchmark` class, combined with `argparse` CLI flags. The only dataclass config is `TransformerConfig` for model architecture parameters.

### Test style
Tests build data inline via helper functions (e.g., `_corpus()`, `_tok()`). Integration tests run actual Lightning training loops with tiny models. Use `tmp_path` for temporary files.
