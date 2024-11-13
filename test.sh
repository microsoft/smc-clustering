# Exit immediately if anything goes wrong
set -e

# Style check
ruff format --check

# Organise imports
ruff check --select I

# Lint -- disable for now
# ruff check

# Type check -- disable for now
# pyright

# Run tests
python3 -m pytest
