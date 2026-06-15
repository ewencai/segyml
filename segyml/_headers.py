"""SEG-Y header parsing: Text, Binary, and Trace headers.

SEG-Y Revision 0 (1975) and Revision 1 (2002) supported.
"""

import struct
from typing import Dict, Any, Optional

# EBCDIC to ASCII translation table (simplified: main printable chars)
_EBCDIC_TO_ASCII = {
    0x40: ' ', 0x4B: '.', 0x4C: '<', 0x4D: '(', 0x4E: '+',
    0x50: '&', 0x5A: '!', 0x5B: '$', 0x5C: '*', 0x5D: ')',
    0x5E: ';', 0x60: '-', 0x61: '/', 0x6B: ',', 0x6C: '%',
    0x6D: '_', 0x6E: '>', 0x6F: '?', 0x79: '`', 0x7A: ':',
    0x7B: '#', 0x7C: '@', 0x7D: "'", 0x7E: '=', 0x7F: '"',
    0x81: 'a', 0x82: 'b', 0x83: 'c', 0x84: 'd', 0x85: 'e',
    0x86: 'f', 0x87: 'g', 0x88: 'h', 0x89: 'i', 0x91: 'j',
    0x92: 'k', 0x93: 'l', 0x94: 'm', 0x95: 'n', 0x96: 'o',
    0x97: 'p', 0x98: 'q', 0x99: 'r', 0xA2: 's', 0xA3: 't',
    0xA4: 'u', 0xA5: 'v', 0xA6: 'w', 0xA7: 'x', 0xA8: 'y',
    0xA9: 'z',
    0xC1: 'A', 0xC2: 'B', 0xC3: 'C', 0xC4: 'D', 0xC5: 'E',
    0xC6: 'F', 0xC7: 'G', 0xC8: 'H', 0xC9: 'I', 0xD1: 'J',
    0xD2: 'K', 0xD3: 'L', 0xD4: 'M', 0xD5: 'N', 0xD6: 'O',
    0xD7: 'P', 0xD8: 'Q', 0xD9: 'R', 0xE2: 'S', 0xE3: 'T',
    0xE4: 'U', 0xE5: 'V', 0xE6: 'W', 0xE7: 'X', 0xE8: 'Y',
    0xE9: 'Z', 0xF0: '0', 0xF1: '1', 0xF2: '2', 0xF3: '3',
    0xF4: '4', 0xF5: '5', 0xF6: '6', 0xF7: '7', 0xF8: '8',
    0xF9: '9',
}

TEXT_HEADER_SIZE = 3200  # bytes (40 lines × 80 chars)
BINARY_HEADER_SIZE = 400  # bytes
TRACE_HEADER_SIZE = 240  # bytes

# Data sample format codes
FORMAT_IBM_FLOAT = 1
FORMAT_INT32 = 2
FORMAT_INT16 = 3
FORMAT_FIXED_POINT = 4
FORMAT_IEEE_FLOAT32 = 5
FORMAT_INT64 = 6  # Rev 1 only
FORMAT_INT8 = 8

FORMAT_BYTES = {
    FORMAT_IBM_FLOAT: 4,
    FORMAT_INT32: 4,
    FORMAT_INT16: 2,
    FORMAT_FIXED_POINT: 4,
    FORMAT_IEEE_FLOAT32: 4,
    FORMAT_INT64: 8,
    FORMAT_INT8: 1,
}

# Trace header byte offsets and formats (key SEG-Y fields)
_TRACE_HEADER_MAP = {
    # (byte_offset, struct_format, name, description)
    0:  ('trace_seq_line', 'Trace sequence number within line'),
    4:  ('trace_seq_file', 'Trace sequence number within file'),
    8:  ('original_field_record', 'Original field record number'),
    12: ('trace_number_field', 'Trace number within field record'),
    20: ('cdp', 'CDP ensemble number'),
    24: ('trace_number_cdp', 'Trace number within CDP ensemble'),
    36: ('source_x', 'Source coordinate X'),
    40: ('source_y', 'Source coordinate Y'),
    72: ('receiver_x', 'Receiver group coordinate X'),
    76: ('receiver_y', 'Receiver group coordinate Y'),
    84: ('group_x', 'Group coordinate X'),
    88: ('group_y', 'Group coordinate Y'),
    108:('dt', 'Sample interval (microseconds)'),
    114:('n_samples', 'Number of samples in this trace'),
    156:('inline', 'Inline number (Rev 1)'),
    160:('crossline', 'Crossline number (Rev 1)'),
    180:('x', 'X coordinate (Rev 1)'),
    188:('y', 'Y coordinate (Rev 1)'),
}


def ebcdic_to_ascii(ebcdic_bytes: bytes) -> str:
    """Convert EBCDIC-encoded bytes to ASCII string."""
    result = []
    for b in ebcdic_bytes:
        result.append(_EBCDIC_TO_ASCII.get(b, '?'))
    return ''.join(result).rstrip()


def parse_text_header(data: bytes) -> str:
    """Parse the 3200-byte textual file header (EBCDIC)."""
    if len(data) < TEXT_HEADER_SIZE:
        raise ValueError(f"Not enough data for text header: {len(data)} bytes < {TEXT_HEADER_SIZE}")
    raw = data[:TEXT_HEADER_SIZE]
    # If it looks like ASCII already, use it directly
    # Check raw bytes: most should be printable ASCII
    if sum(32 <= b <= 126 for b in raw) > len(raw) * 0.9:
        try:
            return raw.decode('ascii').rstrip()
        except UnicodeDecodeError:
            pass
    return ebcdic_to_ascii(raw)


def parse_binary_header(data: bytes) -> Dict[str, Any]:
    """Parse the 400-byte binary file header.

    Returns dict with keys:
        job_id, line_number, reel_number, n_traces_per_ensemble,
        n_aux_traces, dt (sample interval in microseconds),
        dt_original, n_samples, n_samples_original, data_format,
        cdp_fold, trace_sort, measurement_system
    """
    if len(data) < BINARY_HEADER_SIZE:
        raise ValueError(f"Not enough data for binary header: {len(data)} bytes < {BINARY_HEADER_SIZE}")

    h = data[:BINARY_HEADER_SIZE]

    return {
        'job_id': struct.unpack_from('>I', h, 0)[0],
        'line_number': struct.unpack_from('>I', h, 4)[0],
        'reel_number': struct.unpack_from('>I', h, 8)[0],
        'n_traces_per_ensemble': struct.unpack_from('>H', h, 12)[0],
        'n_aux_traces': struct.unpack_from('>H', h, 14)[0],
        'dt': struct.unpack_from('>H', h, 16)[0],  # microseconds
        'dt_original': struct.unpack_from('>H', h, 18)[0],
        'n_samples': struct.unpack_from('>H', h, 20)[0],
        'n_samples_original': struct.unpack_from('>H', h, 22)[0],
        'data_format': struct.unpack_from('>H', h, 24)[0],  # 1=IBM, 5=IEEE, etc.
        'cdp_fold': struct.unpack_from('>H', h, 26)[0],
        'trace_sort': struct.unpack_from('>H', h, 28)[0],
        'measurement_system': struct.unpack_from('>H', h, 34)[0],
        'rev_number': struct.unpack_from('>H', h, 300)[0],  # SEG-Y revision
        'fixed_length_traces': struct.unpack_from('>H', h, 302)[0],
        'n_extended_headers': struct.unpack_from('>H', h, 304)[0],
    }


def parse_trace_header(data: bytes) -> Dict[str, Any]:
    """Parse a 240-byte trace header. Returns dict of key fields."""
    if len(data) < TRACE_HEADER_SIZE:
        raise ValueError(f"Not enough data for trace header: {len(data)} bytes < {TRACE_HEADER_SIZE}")

    h = {}
    for offset, (name, _desc) in _TRACE_HEADER_MAP.items():
        # All standard SEG-Y trace header fields are 4-byte big-endian ints
        h[name] = struct.unpack_from('>i', data, offset)[0]
    return h


def _build_binary_header_bytes(
    dt: int = 4000,
    n_samples: int = 0,
    data_format: int = FORMAT_IEEE_FLOAT32,
    n_traces: int = 0,
) -> bytes:
    """Build a minimal binary header (400 bytes)."""
    buf = bytearray(BINARY_HEADER_SIZE)
    struct.pack_into('>I', buf, 0, 1)              # job_id
    struct.pack_into('>I', buf, 4, 1)              # line_number
    struct.pack_into('>H', buf, 12, n_traces)      # n_traces_per_ensemble
    struct.pack_into('>H', buf, 16, dt)            # sample interval µs
    struct.pack_into('>H', buf, 20, n_samples)     # n_samples
    struct.pack_into('>H', buf, 22, n_samples)     # original n_samples
    struct.pack_into('>H', buf, 24, data_format)   # data format
    struct.pack_into('>H', buf, 302, 1)            # fixed_length_traces
    struct.pack_into('>H', buf, 304, 0)            # no extended headers
    return bytes(buf)


def _build_text_header_bytes(text: Optional[str] = None) -> bytes:
    """Build a 3200-byte textual header. Pads with spaces."""
    buf = bytearray(TEXT_HEADER_SIZE)
    if text:
        encoded = text.encode('ascii', errors='replace')
        buf[:min(len(encoded), TEXT_HEADER_SIZE)] = encoded[:TEXT_HEADER_SIZE]
    else:
        # Fill with spaces
        for i in range(TEXT_HEADER_SIZE):
            buf[i] = 0x20
    for i in range(min(40, len(buf))):
        if i * 80 + 79 < TEXT_HEADER_SIZE:
            buf[i * 80 + 79] = 0x0A  # newline at end of each 80-char line
    return bytes(buf)
