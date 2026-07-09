"""Low-level SEG-Y byte I/O operations."""

import struct
import numpy as np
from typing import Iterator, Tuple, Dict, Any

from ._headers import (
    parse_text_header, parse_binary_header, parse_trace_header,
    _build_text_header_bytes, _build_binary_header_bytes,
    TEXT_HEADER_SIZE, BINARY_HEADER_SIZE, TRACE_HEADER_SIZE,
    FORMAT_IBM_FLOAT, FORMAT_INT32, FORMAT_INT16,
    FORMAT_IEEE_FLOAT32, FORMAT_INT8, FORMAT_BYTES,
)
from ._ibm_float import ibm2float32


def _decode_samples(raw: bytes, fmt_code: int, n_samples: int) -> np.ndarray:
    """Decode raw sample bytes into numpy float32 array.

    Args:
        raw: Raw sample bytes.
        fmt_code: SEG-Y data format code.
        n_samples: Expected number of samples.

    Returns:
        np.ndarray of float32.
    """
    if fmt_code == FORMAT_IBM_FLOAT:
        return ibm2float32(raw)
    elif fmt_code == FORMAT_INT32:
        data = np.frombuffer(raw, dtype='>i4').astype(np.float32)
        if len(data) > n_samples:
            data = data[:n_samples]
        return data
    elif fmt_code == FORMAT_INT16:
        data = np.frombuffer(raw, dtype='>i2').astype(np.float32)
        if len(data) > n_samples:
            data = data[:n_samples]
        return data
    elif fmt_code == FORMAT_IEEE_FLOAT32:
        data = np.frombuffer(raw, dtype='>f4').astype(np.float32)
        if len(data) > n_samples:
            data = data[:n_samples]
        return data
    elif fmt_code == FORMAT_INT8:
        data = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
        if len(data) > n_samples:
            data = data[:n_samples]
        return data
    else:
        raise ValueError(f"Unsupported data format code: {fmt_code}")


def _encode_samples(data: np.ndarray, fmt_code: int) -> bytes:
    """Encode float32 array into sample bytes.

    Args:
        data: numpy float32 array.
        fmt_code: SEG-Y data format code.

    Returns:
        Raw sample bytes.
    """
    data = np.asarray(data, dtype=np.float32).ravel()

    if fmt_code == FORMAT_IBM_FLOAT:
        from ._ibm_float import float2ibm
        return float2ibm(data)
    elif fmt_code == FORMAT_IEEE_FLOAT32:
        return data.astype('>f4').tobytes()
    elif fmt_code == FORMAT_INT32:
        return np.clip(data, -2147483648, 2147483647).astype('>i4').tobytes()
    elif fmt_code == FORMAT_INT16:
        return np.clip(data, -32768, 32767).astype('>i2').tobytes()
    elif fmt_code == FORMAT_INT8:
        return np.clip(data, -128, 127).astype(np.int8).tobytes()
    else:
        raise ValueError(f"Unsupported data format code: {fmt_code}")


def iter_traces(
    path: str,
    trace_indices: slice = slice(None),
) -> Iterator[Tuple[Dict[str, Any], np.ndarray]]:
    """Iterate over traces in a SEG-Y file, yielding (trace_header_dict, data_array).

    Args:
        path: Path to SEG-Y file.
        trace_indices: slice to select specific traces.
    """
    with open(path, 'rb') as f:
        # Read text header
        text_bytes = f.read(TEXT_HEADER_SIZE)
        text_header = parse_text_header(text_bytes)

        # Read binary header
        bin_bytes = f.read(BINARY_HEADER_SIZE)
        bin_header = parse_binary_header(bin_bytes)

        fmt_code = bin_header['data_format']
        n_samples = bin_header['n_samples']
        sample_bytes = FORMAT_BYTES.get(fmt_code, 4)
        trace_bytes = TRACE_HEADER_SIZE + n_samples * sample_bytes

        # Skip extended headers if present (Rev 1)
        n_ext = bin_header.get('n_extended_headers', 0)
        if n_ext > 0:
            f.seek(n_ext * TEXT_HEADER_SIZE, 1)

        trace_num = 0
        while True:
            header_raw = f.read(TRACE_HEADER_SIZE)
            if len(header_raw) < TRACE_HEADER_SIZE:
                break

            data_raw = f.read(n_samples * sample_bytes)
            if len(data_raw) < n_samples * sample_bytes:
                break

            if trace_num >= (trace_indices.start or 0):
                if trace_indices.stop is not None and trace_num >= trace_indices.stop:
                    break
                if trace_indices.step is not None and (trace_num - (trace_indices.start or 0)) % trace_indices.step != 0:
                    trace_num += 1
                    continue

                header = parse_trace_header(header_raw)
                samples = _decode_samples(data_raw, fmt_code, n_samples)
                yield header, samples

            trace_num += 1


def read_segy(path: str) -> Dict[str, Any]:
    """Read an entire SEG-Y file and return headers + data.

    Returns dict with keys:
        text_header, binary_header, traces (list of dicts), data (n_traces × n_samples np.ndarray)
    """
    with open(path, 'rb') as f:
        text_bytes = f.read(TEXT_HEADER_SIZE)
        text_header = parse_text_header(text_bytes)

        bin_bytes = f.read(BINARY_HEADER_SIZE)
        bin_header = parse_binary_header(bin_bytes)

        fmt_code = bin_header['data_format']
        n_samples = bin_header['n_samples']
        sample_bytes = FORMAT_BYTES.get(fmt_code, 4)

        # Skip extended headers
        n_ext = bin_header.get('n_extended_headers', 0)
        if n_ext > 0:
            f.seek(n_ext * TEXT_HEADER_SIZE, 1)

        traces = []
        all_data = []

        while True:
            header_raw = f.read(TRACE_HEADER_SIZE)
            if len(header_raw) < TRACE_HEADER_SIZE:
                break

            data_raw = f.read(n_samples * sample_bytes)
            if len(data_raw) < n_samples * sample_bytes:
                break

            traces.append(parse_trace_header(header_raw))
            all_data.append(_decode_samples(data_raw, fmt_code, n_samples))

        data = np.array(all_data, dtype=np.float32) if all_data else np.array([], dtype=np.float32)

        return {
            'text_header': text_header,
            'binary_header': bin_header,
            'traces': traces,
            'data': data,
        }


def write_segy(
    path: str,
    data: np.ndarray,
    dt: int = 4000,
    data_format: int = FORMAT_IEEE_FLOAT32,
    text_header: str = "",
    trace_headers: list = None,
    inline: np.ndarray = None,
    crossline: np.ndarray = None,
    source_x: np.ndarray = None,
    source_y: np.ndarray = None,
    cdp_x: np.ndarray = None,
    cdp_y: np.ndarray = None,
) -> None:
    """Write a SEG-Y file.

    Args:
        path: Output file path.
        data: 2D numpy array (n_traces, n_samples).
        dt: Sample interval in microseconds.
        data_format: SEG-Y data format code (default: 5 = IEEE float32).
        text_header: Custom text header string.
        trace_headers: List of custom trace header dicts (optional).
        inline, crossline, source_x, source_y, cdp_x, cdp_y: Geometry arrays.
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_traces, n_samples = data.shape

    with open(path, 'wb') as f:
        # Write text header
        f.write(_build_text_header_bytes(text_header if text_header else None))

        # Write binary header
        bin_hdr = _build_binary_header_bytes(
            dt=dt,
            n_samples=n_samples,
            data_format=data_format,
            n_traces=n_traces,
        )
        f.write(bin_hdr)

        # Write traces
        for i in range(n_traces):
            # Build trace header
            hdr = bytearray(TRACE_HEADER_SIZE)
            struct.pack_into('>i', hdr, 0, i + 1)           # trace_seq_line
            struct.pack_into('>i', hdr, 4, i + 1)           # trace_seq_file
            struct.pack_into('>i', hdr, 20, i + 1)          # cdp
            struct.pack_into('>H', hdr, 114, n_samples)     # n_samples (bytes 115-116)
            struct.pack_into('>H', hdr, 116, dt)            # dt, µs (bytes 117-118)

            if inline is not None:
                struct.pack_into('>i', hdr, 156, int(inline[i]))
            if crossline is not None:
                struct.pack_into('>i', hdr, 160, int(crossline[i]))
            if source_x is not None:
                struct.pack_into('>i', hdr, 36, int(source_x[i]))
            if source_y is not None:
                struct.pack_into('>i', hdr, 40, int(source_y[i]))
            if cdp_x is not None:
                struct.pack_into('>i', hdr, 180, int(cdp_x[i]))
            if cdp_y is not None:
                struct.pack_into('>i', hdr, 188, int(cdp_y[i]))

            # Merge custom trace headers if provided
            if trace_headers and i < len(trace_headers):
                from ._headers import _TRACE_HEADER_MAP
                for key, val in trace_headers[i].items():
                    for offset, (name, fmt, _) in _TRACE_HEADER_MAP.items():
                        if name == key:
                            struct.pack_into(fmt, hdr, offset, int(val))
                            break

            f.write(bytes(hdr))
            f.write(_encode_samples(data[i], data_format))
