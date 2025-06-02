from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pylops
import segyio
from fastmcp import FastMCP, Image
from pylops.basicoperators import Restriction
from pylops.optimization.sparsity import fista
from pylops.signalprocessing import FFT2D
from scipy.signal import butter, filtfilt
from skimage.restoration import denoise_tv_chambolle

mcp = FastMCP(
    name="Atomic",
    instructions="""
        This server provides an automatic workflow of seismic processing from data loading, visualization, processing!
        """,
)


@mcp.resource("dataref://datawithnoise")
def evaluate_is_data_noisy() -> Image:
    """This is an example of a noisy data image. If you see that the data has characteristics like this, you can use the denoise function to remove noise."""
    path = Path(__file__).parent / "dataref" / "datawithnoise.png"
    return Image(path=path, format="png")


@mcp.resource("dataref://datawithgap")
def evaluate_is_data_with_gap() -> Image:
    """This is an example of a data with gaps. If you see that the data has characteristics like this, you can use the interpolate_gap function to fill the gaps."""
    path = Path(__file__).parent / "dataref" / "datawithgap.png"
    return Image(path=path, format="png")


@mcp.tool()
def list_available_segy_files() -> str:
    """This functions returns all available .segy files on the system."""
    data_directory = Path(__file__).parent / "data"
    files = ", ".join([str(path) for path in data_directory.glob("*.sgy")])
    print(f"{data_directory=}, {files=}")
    return files


@mcp.tool()
def list_available_numpy_files() -> str:
    """This functions returns all available .npz files on the system."""
    data_directory = Path(__file__).parent / "data"
    files = ", ".join([str(path) for path in data_directory.glob("*.npz")])
    print(f"{data_directory=}, {files=}")
    return files


@mcp.tool()
def open_seismic_data(file_path: str) -> str:
    """Always use this function to open a SEG-Y file at the beginning of the chat. To make the data available for other functions."""
    print(f"open_seismic_data {file_path=}")
    try:
        f = segyio.open(file_path, "rb", ignore_geometry=True)
        data = segyio.collect(f.trace)
        data = data[::4, :].reshape((240, 240, 2000))
        new_file_path = file_path.replace(".sgy", ".npz")
        np.savez(new_file_path, data=data)
        return new_file_path
    except Exception as e:
        return f"Error opening file: {str(e)}"


@mcp.tool()
def read_seismic_info(file_path: str) -> str:
    """Read the seismic information from a SEG-Y file."""
    print(f"read_seismic_info {file_path=}")
    with segyio.open(file_path, "rb", ignore_geometry=True) as f:
        return segyio.tools.wrap(f.text[0])


@mcp.tool()
def read_seismic_binary_info(file_path: str) -> str:
    """Read seismic binary header information from a SEG-Y file."""
    try:
        with segyio.open(file_path, "rb", ignore_geometry=True) as f:
            header = f.bin  # Read the first trace header
            return header
    except Exception as e:
        return f"Error reading header: {str(e)}"


@mcp.tool()
def read_trace_header(
    file_path: str,
) -> str:
    """Read the seismic trace header information from a SEG-Y file."""
    try:
        with segyio.open(file_path, "rb", ignore_geometry=True) as f:
            header = f.header[0]  # Read the first trace header
            return header
    except Exception as e:
        return f"Error reading header: {str(e)}"


@mcp.tool()
def open_and_visualize_2d(filepath: str, shot: int, title: str) -> Image:
    """Open and Visualize a 2D seismic section.

    The input numpy file can either be 2D or 3D.

    Args:
        filepath (str): Path to the npz file of the data.
        shot (int): The index of the shot to visualize.
        title (str): The title for the visualization. You can name it relevant to the data.

    Returns:
        Image: An image object containing the visualized seismic data.
    """

    data = np.load(filepath)["data"]
    jpeg_path = f"{filepath.removesuffix('.npz')}_{shot}.jpeg"
    print(f"{data=}, {jpeg_path=}")
    plt.figure(figsize=(8, 6))

    plt.figure(figsize=(8, 6))
    if len(data.shape) > 2:
        plt.imshow(data[:, shot, :].T, aspect="auto", cmap="gray", vmin=-1e5, vmax=1e5)
    else:
        plt.imshow(data.T, aspect="auto", cmap="gray", vmin=-1e5, vmax=1e5)

    # Title uses the correct filename of the SEG-Y file
    plt.title(title)
    plt.xlabel("Xline")
    plt.ylabel("Samples")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(jpeg_path, format="jpeg")
    plt.close()
    return Image(path=jpeg_path, format="jpeg")


@mcp.tool()
def butter_bandpass_filter(filepath: str, lowcut: int, highcut: int, shot: int) -> str:
    r"""Apply Butterworth bandpass filter

    Apply Butterworth bandpass filter over time axis of input data

    In and output are 3D numpy arrays.

    Args:
        filepath : Path to the npz file containing the data to be filtered.
        data : 1D or 2D array where filtering is applied along the last axis
        lowcut : Lower cut-off frequency
        highcut : Upper cut-off frequency

    Returns:
        The new path of the filtered data.
    """
    fs = 1 / 4000
    data = np.load(filepath)["data"]
    new_filepath = filepath.replace(".npz", f"{shot}_filtered.npz")
    data = data[:, shot, :]
    order = 5
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="bandpass", analog=False)
    y = filtfilt(b, a, data, axis=-1)
    y = np.expand_dims(y, axis=1)
    np.savez(new_filepath, data=y)
    return new_filepath


@mcp.tool()
def interpolate_gap(filepath: str) -> str:
    """Interpolate gaps in the seismic data using sparsity promoting inversion.

    In and output are 2D numpy arrays.

    Args:
        filepath: This is the path to the .npz file with the data.

    Returns:
        The new path of the interpolated data.
    """
    print(f"{filepath=}")
    new_filepath = filepath.replace(".npz", "_interpolated.npz")
    nfft_x, nfft_t = 1024, 1024

    data_miss = np.load(filepath)["data"]

    zero_rows = np.where(np.all(data_miss == 0, axis=1))[0]
    all_indices = np.arange(data_miss.shape[0])
    available_trace_indices = np.setdiff1d(all_indices, zero_rows)

    Rop = Restriction(dims=(data_miss.shape), iava=available_trace_indices, axis=0, dtype="float64")
    Fop = FFT2D(dims=(data_miss.shape), nffts=(nfft_x, nfft_t), dtype=np.complex128)

    F0op = Rop * Fop.H
    data0 = np.delete(data_miss, zero_rows, axis=0).ravel()

    with pylops.disabled_ndarray_multiplication():
        pinv0, _, _ = fista(F0op, data0, niter=100, eps=1000, eigsdict=dict(niter=5, tol=1e-2), show=True)
    data_recover = np.real(Fop.H * pinv0).reshape(data_miss.shape)
    np.savez(new_filepath, data=data_recover)
    return new_filepath


@mcp.tool()
def denoise(filepath: str) -> str:
    """Denoise seismic data using TV Regularization.

    In and output are 2D numpy arrays.

    Args:
        filepath (str): Path to the npz file of the data.

    Returns:
        The new path of the denoised data.
    """
    new_filepath = filepath.replace(".npz", "_denoised.npz")
    data = np.load(filepath)["data"]
    image_denoise = denoise_tv_chambolle(data, weight=100000)
    np.savez(new_filepath, data=image_denoise)
    return new_filepath


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=9000,
        log_level="debug",
        path="/atomic",
    )
