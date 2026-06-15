"""Tests for segyml IBM float conversion."""

import numpy as np
import pytest
from segyml._ibm_float import ibm2float32, float2ibm


class TestIBMFloat:
    def test_roundtrip_ieee(self):
        """IEEE float values should survive IBM float roundtrip."""
        original = np.array([0.0, 1.0, -1.0, 3.14159, -2.71828,
                             1000.0, -0.001, 1e6, -1e-6], dtype=np.float32)
        ibm_bytes = float2ibm(original)
        recovered = ibm2float32(ibm_bytes)
        # IBM float has ~6-7 decimal digits of precision
        np.testing.assert_allclose(original, recovered, rtol=1e-5, atol=1e-6)

    def test_zero_array(self):
        """Zero array should roundtrip exactly."""
        original = np.zeros(100, dtype=np.float32)
        ibm_bytes = float2ibm(original)
        recovered = ibm2float32(ibm_bytes)
        np.testing.assert_array_equal(original, recovered)

    def test_large_array(self):
        """Large array should be handled efficiently."""
        original = np.random.randn(10000).astype(np.float32)
        ibm_bytes = float2ibm(original)
        assert len(ibm_bytes) == 10000 * 4
        recovered = ibm2float32(ibm_bytes)
        np.testing.assert_allclose(original, recovered, rtol=1e-5, atol=1e-6)

    def test_powers_of_two(self):
        """Powers of two should roundtrip exactly (exact in both IEEE and IBM)."""
        original = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
                             0.25, 0.125, -0.5, -2.0, -8.0], dtype=np.float32)
        ibm_bytes = float2ibm(original)
        recovered = ibm2float32(ibm_bytes)
        np.testing.assert_array_almost_equal(original, recovered)
