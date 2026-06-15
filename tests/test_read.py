"""Tests for segyml I/O: read/write roundtrip."""

import os
import tempfile
import numpy as np
import pytest
import segyml
from segyml._headers import FORMAT_IBM_FLOAT, FORMAT_IEEE_FLOAT32


class TestRoundtrip:
    def test_write_read_ieee(self):
        """Write IEEE float SEG-Y, read back, data should match."""
        n_traces, n_samples = 10, 256
        data = np.random.randn(n_traces, n_samples).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.segy', delete=False) as f:
            tmp_path = f.name

        try:
            segyml.save(tmp_path, data, dt=4000, data_format=FORMAT_IEEE_FLOAT32)
            loaded, headers = segyml.load(tmp_path)

            assert loaded.shape == data.shape
            np.testing.assert_array_equal(loaded, data)
            assert headers['binary_header']['dt'] == 4000
            assert headers['binary_header']['data_format'] == FORMAT_IEEE_FLOAT32
            assert len(headers['traces']) == n_traces
        finally:
            os.unlink(tmp_path)

    def test_write_read_ibm(self):
        """Write IBM float SEG-Y, read back, data should match within tolerance."""
        n_traces, n_samples = 10, 128
        data = np.random.randn(n_traces, n_samples).astype(np.float32) * 100

        with tempfile.NamedTemporaryFile(suffix='.segy', delete=False) as f:
            tmp_path = f.name

        try:
            segyml.save(tmp_path, data, dt=2000, data_format=FORMAT_IBM_FLOAT)
            loaded, headers = segyml.load(tmp_path)

            assert loaded.shape == data.shape
            np.testing.assert_allclose(loaded, data, rtol=1e-5, atol=1e-5)
            assert headers['binary_header']['data_format'] == FORMAT_IBM_FLOAT
        finally:
            os.unlink(tmp_path)

    def test_single_trace(self):
        """Single trace write/read should work."""
        data = np.random.randn(512).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.segy', delete=False) as f:
            tmp_path = f.name

        try:
            segyml.save(tmp_path, data)
            loaded, headers = segyml.load(tmp_path)

            assert loaded.shape == (1, 512)
            np.testing.assert_array_equal(loaded[0], data)
        finally:
            os.unlink(tmp_path)

    def test_geometry_headers(self):
        """Inline/crossline headers should be preserved."""
        n_traces = 5
        data = np.random.randn(n_traces, 100).astype(np.float32)
        inlines = np.array([100, 200, 300, 400, 500])
        xlines = np.array([10, 20, 30, 40, 50])

        with tempfile.NamedTemporaryFile(suffix='.segy', delete=False) as f:
            tmp_path = f.name

        try:
            segyml.save(tmp_path, data, inline=inlines, crossline=xlines)
            loaded, headers = segyml.load(tmp_path)

            for i, tr in enumerate(headers['traces']):
                assert tr['inline'] == inlines[i]
                assert tr['crossline'] == xlines[i]
        finally:
            os.unlink(tmp_path)


class TestTraceSelection:
    def test_slice_traces(self):
        """Selecting traces by slice should work."""
        n_traces = 50
        data = np.random.randn(n_traces, 100).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.segy', delete=False) as f:
            tmp_path = f.name

        try:
            segyml.save(tmp_path, data)
            loaded, headers = segyml.load(tmp_path, traces=slice(10, 20))
            assert loaded.shape == (10, 100)
            np.testing.assert_array_equal(loaded, data[10:20])
        finally:
            os.unlink(tmp_path)

    def test_list_traces(self):
        """Selecting specific trace indices should work."""
        n_traces = 20
        data = np.arange(n_traces * 50, dtype=np.float32).reshape(n_traces, 50)

        with tempfile.NamedTemporaryFile(suffix='.segy', delete=False) as f:
            tmp_path = f.name

        try:
            segyml.save(tmp_path, data)
            loaded, headers = segyml.load(tmp_path, traces=[0, 5, 10, 15])
            assert loaded.shape == (4, 50)
            np.testing.assert_array_equal(loaded[0], data[0])
            np.testing.assert_array_equal(loaded[1], data[5])
            np.testing.assert_array_equal(loaded[2], data[10])
            np.testing.assert_array_equal(loaded[3], data[15])
        finally:
            os.unlink(tmp_path)
