# segyml — Python SEG-Y Library with ML Integration

[![PyPI](https://img.shields.io/badge/pypi-segyml-blue)](https://pypi.org/project/segyml/)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-GPL%20v2-green)](./LICENSE)

**segyml** is a lightweight, fast Python library for reading, writing, and converting SEG-Y seismic data files. Built for the AI era: load seismic data directly into PyTorch tensors with one line of code.

## Why segyml?

- **Zero-friction ML pipeline**: `load("survey.segy", backend="torch")` → ready for training
- **Minimal dependencies**: Only `numpy` required. `torch` and `matplotlib` are optional
- **Familiar API**: Designed like `numpy.load` — you already know how to use it
- **Fast**: Vectorized IBM float conversion, streaming reads for large files

## Installation

```bash
pip install segyml

# With PyTorch support
pip install segyml[torch]

# With visualization
pip install segyml[viz]

# Everything
pip install segyml[all]
```

## Quick Start

```python
import segyml

# Load SEG-Y as numpy array
data, headers = segyml.load("survey.segy")
print(f"Traces: {data.shape[0]}, Samples: {data.shape[1]}")
print(f"Sample interval: {headers['binary_header']['dt']} µs")

# Load directly as PyTorch tensor
tensor, headers = segyml.load("survey.segy", backend="torch")
# → torch.Tensor, ready for your neural network

# Write SEG-Y
segyml.save(data, "output.segy", dt=4000)

# Batch convert ASC files to SEG-Y
segyml.asc2segy("X:/raw_data/", "merged.segy")

# Visualize
segyml.wiggle(data[:, :50], dt=0.004, save_path="section.png")
```

## Supported Formats

| SEG-Y Feature | Support |
|---|---|
| Revision 0 (1975) | ✅ |
| Revision 1 (2002) | ✅ |
| IBM Float (format 1) | ✅ |
| IEEE Float32 (format 5) | ✅ |
| Int32 (format 2) | ✅ |
| Int16 (format 3) | ✅ |
| Int8 (format 8) | ✅ |
| EBCDIC headers | ✅ |
| 3D geometry (inline/crossline) | ✅ |

## API Reference

### `segyml.load(path, traces=None, backend="numpy")`
Load a SEG-Y file.
- `traces`: `slice(0, 100)` for first 100 traces, or `[1,5,10]` for specific indices
- `backend`: `"numpy"` or `"torch"`

### `segyml.save(path, data, dt=4000, data_format=5, ...)`
Save data as SEG-Y. Supports geometry headers.

### `segyml.asc2segy(asc_dir, output_path, dt=4000)`
Batch convert ASCII text files to SEG-Y.

### `segyml.wiggle(data, dt=0.004, save_path=None)`
Wiggle trace plot with variable-area fill.

### `segyml.image(data, cmap="gray", save_path=None)`
2D seismic image plot.

## Architecture

```
segyml/
├── api.py          → load(), save(), asc2segy()
├── _io.py          → Low-level byte I/O, streaming
├── _headers.py     → Text/Binary/Trace header parsing
├── _ibm_float.py   → IBM float ↔ IEEE 754 conversion
├── tensor.py       → PyTorch integration
└── visualize.py    → matplotlib plots
```

## Roadmap

- [ ] SAC format support
- [ ] TensorFlow backend
- [ ] Dask support for out-of-core processing
- [ ] GPU-accelerated IBM float conversion

## License

GPL v2 — see [LICENSE](./LICENSE)

---

Built by [Ewen Cai](https://github.com/ewencai). Based on decades of SEG-Y processing expertise from the field.
