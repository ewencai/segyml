"""PyTorch tensor integration for segyml.

Lightweight wrapper — torch is an optional dependency.
"""

import numpy as np

_HAS_TORCH = False
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None


def to_tensor(data: np.ndarray, dtype=None, device: str = "cpu") -> "torch.Tensor":
    """Convert numpy array to torch.Tensor with zero-copy when possible.

    Args:
        data: numpy array (typically from segyml.load()).
        dtype: torch dtype (default: torch.float32).
        device: Target device.

    Returns:
        torch.Tensor sharing memory with the numpy array when on CPU.
    """
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for tensor output. "
            "Install with: pip install segyml[torch]"
        )
    if dtype is None:
        dtype = torch.float32
    return torch.from_numpy(np.asarray(data)).to(dtype=dtype, device=device)


def to_numpy(tensor: "torch.Tensor") -> np.ndarray:
    """Convert torch.Tensor to numpy array.

    Args:
        tensor: PyTorch tensor.

    Returns:
        numpy float32 array.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required. Install with: pip install segyml[torch]")
    return tensor.detach().cpu().numpy().astype(np.float32)
