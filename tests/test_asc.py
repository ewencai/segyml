"""Tests for segyml ASC to SEG-Y conversion."""

import os
import tempfile
import numpy as np
import segyml


class TestASC2SEGY:
    def test_basic_conversion(self):
        """Basic ASC to SEG-Y conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a few ASC files
            for i in range(5):
                asc_path = os.path.join(tmpdir, f"trace_{i:04d}.asc")
                values = np.sin(np.linspace(0, 4 * np.pi, 200)) * (i + 1)
                np.savetxt(asc_path, values, fmt='%.6f')

            output = os.path.join(tmpdir, "output.segy")
            n = segyml.asc2segy(tmpdir, output)

            assert n == 5
            data, headers = segyml.load(output)
            assert data.shape == (5, 200)
            # Each trace should have different amplitude
            assert not np.allclose(data[0], data[4])
