# Research on SMC and Diffusion Models for Entity Linking

This repository contains the code and experiments for research on entity linking using SMC and diffusion models.

The work is done in collaboration between Microsoft Research and Lancaster University. 

The recommended way to install the code is to clone the repository, clone DiSK and MS-KeBAB repositories, and install
everything via `uv`.

1. Create the virtual environment and install the package:

   ```bash
   uv sync --locked
   ```

2. Install DiSK (assuming DiSK is cloned in the parent directory of this repository):
   ```bash
   uv pip install -e ../DiSK
   ```
   
   Install DiSK's extra dependencies.
   With CUDA support (12.6 or higher):

   ```bash
   uv pip install -r ../DiSK/requirements.txt -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
   ```

   Without CUDA support:

   ```bash
   uv pip install -r ../DiSK/requirements.txt -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
   ```

3. Install MS-KeBAB package:
   Assuming MS-KEBAB project is cloned in the parent directory of this repository, you can install it as follows:
   
   ```bash
   uv pip install --no-deps -e ../MS-KeBAB
   ```

To sidestep some of the issues with CUDA memory allocation, you can set the following environment variables:
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```