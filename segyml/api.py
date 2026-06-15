"""High-level API for SEG-Y file operations."""

import glob
import os
import numpy as np
from typing import Optional, Union, List

from ._io import read_segy, write_segy, iter_traces
from ._headers import FORMAT_IEEE_FLOAT32, FORMAT_IBM_FLOAT


def load(
    path: str,
    traces: Union[slice, List[int], None] = None,
    backend: str = "numpy",
) -> tuple:
    """Load a SEG-Y file.

    Args:
        path: Path to SEG-Y file.
        traces: Optional trace selection (slice or list of indices).
        backend: "numpy" for np.ndarray, "torch" for torch.Tensor.

    Returns:
        (data, headers) tuple.
        data: shape (n_traces, n_samples)
        headers: dict with keys 'text_header', 'binary_header', 'traces'

    Examples:
        >>> data, hdr = segyml.load("survey.segy")
        >>> data, hdr = segyml.load("survey.segy", traces=slice(0, 100))
        >>> tensor, hdr = segyml.load("survey.segy", backend="torch")
    """
    if traces is None:
        result = read_segy(path)
        data = result['data']
        headers = {
            'text_header': result['text_header'],
            'binary_header': result['binary_header'],
            'traces': result['traces'],
        }
    elif isinstance(traces, slice):
        all_headers = []
        all_data = []
        for hdr, dat in iter_traces(path, trace_indices=traces):
            all_headers.append(hdr)
            all_data.append(dat)
        data = np.array(all_data, dtype=np.float32) if all_data else np.array([], dtype=np.float32)
        headers = {'traces': all_headers}
    elif isinstance(traces, list):
        # Read specific trace indices
        all_headers = []
        all_data = []
        trace_set = set(traces)
        for i, (hdr, dat) in enumerate(iter_traces(path)):
            if i in trace_set:
                all_headers.append(hdr)
                all_data.append(dat)
        data = np.array(all_data, dtype=np.float32) if all_data else np.array([], dtype=np.float32)
        headers = {'traces': all_headers}
    else:
        raise TypeError(f"traces must be slice, list, or None, got {type(traces)}")

    if backend == "torch":
        from .tensor import to_tensor
        data = to_tensor(data)
    elif backend != "numpy":
        raise ValueError(f"Unknown backend: {backend}. Use 'numpy' or 'torch'.")

    return data, headers


def save(
    path: str,
    data: "Union[np.ndarray, 'torch.Tensor']",
    dt: int = 4000,
    data_format: int = FORMAT_IEEE_FLOAT32,
    text_header: str = "",
    inline: Optional[np.ndarray] = None,
    crossline: Optional[np.ndarray] = None,
    source_x: Optional[np.ndarray] = None,
    source_y: Optional[np.ndarray] = None,
    cdp_x: Optional[np.ndarray] = None,
    cdp_y: Optional[np.ndarray] = None,
) -> None:
    """Save data to a SEG-Y file.

    Args:
        path: Output file path.
        data: 2D array (n_traces, n_samples). Supports numpy or torch.
        dt: Sample interval in microseconds.
        data_format: 1=IBM float, 5=IEEE float32 (default).
        text_header: Custom EBCDIC text header string.
        inline/crossline: Geometry arrays, one per trace.
        source_x/source_y/cdp_x/cdp_y: Coordinate arrays.

    Example:
        >>> segyml.save("output.segy", data, dt=4000)
        >>> segyml.save("output.segy", data, dt=2000,
        ...             inline=inlines, crossline=xlines)
    """
    # Convert torch tensor to numpy if needed
    if hasattr(data, 'numpy'):
        from .tensor import to_numpy
        data = to_numpy(data)

    write_segy(
        path=path,
        data=data,
        dt=dt,
        data_format=data_format,
        text_header=text_header,
        inline=inline,
        crossline=crossline,
        source_x=source_x,
        source_y=source_y,
        cdp_x=cdp_x,
        cdp_y=cdp_y,
    )


def asc2segy(
    asc_dir: str,
    output_path: str,
    dt: int = 4000,
    encoding: str = "utf-8",
    delimiter: str = None,
) -> int:
    """Convert ASC text files to SEG-Y format.

    Reads all .asc files in a directory, concatenates them into one SEG-Y file.
    Each ASC file is treated as one trace (single column).

    Args:
        asc_dir: Directory containing .asc files.
        output_path: Output SEG-Y file path.
        dt: Sample interval in microseconds.
        encoding: File encoding (default: utf-8).
        delimiter: Column delimiter (auto-detect if None).

    Returns:
        Number of traces converted.

    Example:
        >>> n = segyml.asc2segy("X:/raw_data/", "output.segy")
        >>> print(f"Converted {n} traces")
    """
    pattern = os.path.join(asc_dir, "*.asc")
    asc_files = sorted(glob.glob(pattern))

    if not asc_files:
        raise FileNotFoundError(f"No .asc files found in {asc_dir}")

    all_data = []
    for fpath in asc_files:
        data = np.loadtxt(fpath, delimiter=delimiter, encoding=encoding)
        if data.ndim == 0:
            data = np.array([data])
        all_data.append(data.ravel().astype(np.float32))

    if not all_data:
        return 0

    # Pad to uniform length
    max_len = max(len(d) for d in all_data)
    padded = np.zeros((len(all_data), max_len), dtype=np.float32)
    for i, d in enumerate(all_data):
        padded[i, :len(d)] = d

    write_segy(
        path=output_path,
        data=padded,
        dt=dt,
        data_format=FORMAT_IEEE_FLOAT32,
    )

    return len(all_data)
