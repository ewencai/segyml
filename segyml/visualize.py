"""Seismic data visualization using matplotlib."""

import numpy as np

_HAS_MPL = False
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    plt = None


def wiggle(
    data: np.ndarray,
    dt: float = 0.004,
    title: str = "",
    xlabel: str = "Trace",
    ylabel: str = "Time (s)",
    skip: int = 1,
    scale: float = 1.0,
    fill: bool = True,
    cmap_name: str = "RdBu",
    save_path: str = None,
    figsize: tuple = (12, 8),
    **kwargs,
):
    """Plot seismic wiggle traces with variable-area fill.

    Args:
        data: 2D array (n_traces, n_samples).
        dt: Sample interval in seconds.
        title: Plot title.
        xlabel, ylabel: Axis labels.
        skip: Plot every Nth trace (for large datasets).
        scale: Amplitude scaling factor.
        fill: Enable variable-area fill (positive amplitudes filled).
        cmap_name: Colormap for fill colors.
        save_path: If provided, save figure to this path.
        figsize: Figure size in inches.

    Example:
        >>> data, _ = segyml.load("survey.segy")
        >>> segyml.wiggle(data[:, :100], dt=0.004, save_path="section.png")
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install segyml[viz]"
        )

    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_traces, n_samples = data.shape
    time_axis = np.arange(n_samples) * dt
    traces_to_plot = list(range(0, n_traces, skip))

    fig, ax = plt.subplots(figsize=figsize)

    for trace_idx, i in enumerate(traces_to_plot):
        trace = data[i] * scale
        offset = trace_idx * 2  # spacing between traces

        if fill:
            # Positive fill
            ax.fill_betweenx(
                time_axis, offset, offset + trace,
                where=trace > 0,
                color='black', alpha=0.6, linewidth=0,
            )
            # Negative fill
            ax.fill_betweenx(
                time_axis, offset, offset + trace,
                where=trace <= 0,
                color='red', alpha=0.3, linewidth=0,
            )

        # Wiggle line
        ax.plot(offset + trace, time_axis, 'k-', linewidth=0.5)

    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def image(
    data: np.ndarray,
    dt: float = 0.004,
    title: str = "",
    cmap: str = "gray",
    save_path: str = None,
    figsize: tuple = (12, 8),
    **kwargs,
):
    """Plot seismic data as a 2D image (grayscale by default).

    Args:
        data: 2D array (n_traces, n_samples).
        dt: Sample interval in seconds.
        title: Plot title.
        cmap: Matplotlib colormap name.
        save_path: Save figure to this path.
        figsize: Figure size in inches.
    """
    if not _HAS_MPL:
        raise ImportError(
            "matplotlib is required. Install with: pip install segyml[viz]"
        )

    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_traces, n_samples = data.shape
    time_axis = np.arange(n_samples) * dt

    fig, ax = plt.subplots(figsize=figsize)
    extent = [0, n_traces, time_axis[-1], time_axis[0]]
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, extent=extent, **kwargs)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time (s)")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Amplitude")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
