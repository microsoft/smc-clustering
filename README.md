# Research in SMC clustering methods

This repository contains the code and experiments for research on entity linking using SMC and diffusion models.

The work is done in collaboration between Microsoft Research and Lancaster University.

The recommended way to install the code is to clone the repository and install everything via `uv`.

1. Create the virtual environment and install the package:

   ```bash
   uv sync --locked
   ```

2. Install MS-KeBAB package:
   Assuming MS-KEBAB project is cloned in the parent directory of this repository, you can install it as follows:

   ```bash
   uv pip install --no-deps -e ../MS-KeBAB
   ```
