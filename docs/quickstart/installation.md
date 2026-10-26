# Install

Requires Python ≥ 3.10 and PyTorch ≥ 2.1. Install in a fresh environment:

```bash
git clone https://github.com/tatsumonkman47/ocean-sda-starter.git && cd ocean-sda-starter
python -m venv .venv && source .venv/bin/activate
pip install -e '.[examples,dev]'
python -m pytest              # ~2 min on CPU
```

Import the package as `obottle_starter`. Keep the repository in place after
installation: the code locates `data/` relative to it.

## Tested environment

**Python 3.13, PyTorch 2.14.1, and NVIDIA L40S**
(2026-10-26). To use the dependency versions tested with all four notebooks:

```bash
pip install -r requirements-tested.txt && pip install -e '.[examples,dev]'
```

Its PyTorch wheel bundles CUDA 13.0 and needs a recent NVIDIA
driver. If `torch.cuda.is_available()` is `False` on your GPU machine,
install the PyTorch build for your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/) and then the
package. The tests run on CPU.

---

[Documentation index](../README.md) · [Project README](../../README.md)
