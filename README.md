# intro_to_jax

Introduction to JAX with CUDA support.

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA 12.9+
- cuDNN support

## Installation

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install JAX with CUDA support directly:
```bash
pip install --upgrade "jax[cuda12]"
```

3. Verify GPU support:
```bash
python -c "import jax; print('Devices:', jax.devices()); print('Backend:', jax.default_backend())"
```

You should see `[cuda(id=0)]` indicating GPU support is enabled.

## Usage

Run the example script:
```bash
python scripts/issue1.py
```
