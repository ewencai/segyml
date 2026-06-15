"""IBM floating-point ↔ IEEE 754 conversion.

SEG-Y format uses IBM System/360 floating-point format (format code 1).
IBM float: 1 sign bit + 7 exponent bits (base 16) + 24 mantissa bits = 4 bytes.

References:
    - https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point
"""

import numpy as np


def ibm2float32(ibm_bytes: bytes) -> np.ndarray:
    """Convert IBM floating-point bytes to IEEE 754 float32.

    Args:
        ibm_bytes: Raw bytes in IBM float format (length must be multiple of 4).

    Returns:
        np.ndarray of np.float32 values.
    """
    if len(ibm_bytes) % 4 != 0:
        raise ValueError(f"IBM float bytes length must be multiple of 4, got {len(ibm_bytes)}")

    arr = np.frombuffer(ibm_bytes, dtype=np.uint8).reshape(-1, 4).astype(np.uint32)
    # Big-endian: byte 0 is most significant
    word = (arr[:, 0].astype(np.uint32) << 24) | \
           (arr[:, 1].astype(np.uint32) << 16) | \
           (arr[:, 2].astype(np.uint32) << 8) | \
           arr[:, 3].astype(np.uint32)

    sign = (word >> 31) & 1
    exponent = ((word >> 24) & 0x7F).astype(np.int32)  # 7-bit characteristic
    mantissa = (word & 0x00FFFFFF).astype(np.float64)

    # IBM float: value = sign * mantissa * 16^(exponent - 64)
    fraction = mantissa / (2.0 ** 24)
    result = (1.0 - 2.0 * sign) * fraction * (16.0 ** (exponent - 64))

    # Handle zero (mantissa == 0)
    result[mantissa == 0] = 0.0

    return result.astype(np.float32)


def float2ibm(data: np.ndarray) -> bytes:
    """Convert IEEE 754 float32 to IBM floating-point bytes.

    Args:
        data: numpy array of float values.

    Returns:
        Raw bytes in IBM float format.
    """
    data = np.asarray(data, dtype=np.float64).ravel()
    sign = np.zeros(len(data), dtype=np.uint32)
    sign[data < 0] = 1

    abs_data = np.abs(data)
    result = np.zeros(len(data), dtype=np.uint32)

    nonzero = abs_data > 0
    if np.any(nonzero):
        # exponent: find power of 16
        exponent = np.zeros(len(data), dtype=np.int32)
        exponent[nonzero] = np.floor(np.log(abs_data[nonzero]) / np.log(16.0)) + 1
        exponent = np.clip(exponent + 64, 0, 127)

        # mantissa: mantissa = abs_val * 2^24 / 16^(unbiased_exponent)
        mantissa = np.zeros(len(data), dtype=np.float64)
        unbiased_exp = exponent[nonzero].astype(np.float64) - 64
        mantissa[nonzero] = abs_data[nonzero] * (2.0 ** 24) / (16.0 ** unbiased_exp)
        mantissa_i = mantissa.astype(np.uint32)

        # Handle mantissa overflow (>= 2^24): divide by 16, increment exponent
        overflow = mantissa_i >= 0x1000000
        while np.any(overflow):
            exponent[overflow] += 1
            mantissa[overflow] /= 16.0
            mantissa_i[overflow] = mantissa[overflow].astype(np.uint32)
            overflow = mantissa_i >= 0x1000000

        mantissa_i = np.clip(mantissa_i, 0, 0xFFFFFF)

        result[nonzero] = (sign[nonzero] << 31) | \
                          (exponent[nonzero].astype(np.uint32) << 24) | \
                          mantissa_i[nonzero]

    # Pack as big-endian bytes
    out = np.zeros(len(data) * 4, dtype=np.uint8)
    out[0::4] = (result >> 24).astype(np.uint8)
    out[1::4] = (result >> 16).astype(np.uint8)
    out[2::4] = (result >> 8).astype(np.uint8)
    out[3::4] = result.astype(np.uint8)

    return out.tobytes()
