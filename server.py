import tempfile

import matplotlib.pyplot as plt

# from pydantic_ai import Agent
import numpy as np
from scipy.signal import butter, filtfilt
from pylops.basicoperators import *
from pylops.optimization.sparsity import fista
from pylops.basicoperators import Diagonal, Restriction
from pylops.signalprocessing import FFT2D
import pylops
import segyio
import skimage
from skimage.restoration import denoise_tv_chambolle
from fastmcp import FastMCP, Image

mcp = FastMCP(
    name="Atomic",
    instructions="""
        This server provides an automatic workflow of seismic processing from data loading, visualization, processing!
        """,
)

@mcp.resource("dataref://datawithnoise")
def evaluate_is_data_noisy()->Image:
    """This is an example of a noisy data image. If you see that the data has characteristics like this, you can use the denoise function to remove noise."""
    return Image(path=r"D:\Jupyter root\NewAtomic\dataref\datawithnoise.png", format="png")
@mcp.resource("dataref://datawithgap")
def evaluate_is_data_with_gap()->Image:
    """This is an example of a data with gaps. If you see that the data has characteristics like this, you can use the interpolate_gap function to fill the gaps."""
    return Image(path=r"D:\Jupyter root\NewAtomic\dataref\datawithgap.png", format="png")

@mcp.tool()
def open_and_visualize_2d(filepath: str, shot: int, title: str) -> Image:
    """Open and Visualize a 2D seismic section.

    Args:
        filepath (str): Path to the npz file of the data.
        shot (int): The index of the shot to visualize.
        title (str): The title for the visualization. You can name it relevant to the data.

    Returns:
        Image: An image object containing the visualized seismic data.

    Notes:
        You need to provide a 2D list. You can use the `get_seismic_data` function to obtain the data.
    """

    data = np.load(filepath)["data"]

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
    plt.savefig("result/"+title + ".png", format="png")
    plt.close()
    jpeg_path = "result/"+ title + ".png"

    return Image(path=jpeg_path, format="png")

# @mcp.tool()
# def open_seismic_data(file_path: str) -> str:
#     """Use this function to open a SEG-Y file and convert it to a numpy array. If the data is already in numpy format, it will save it as a .npz file."""
#     try:
#         f = segyio.open(file_path, "rb", ignore_geometry=True)
#         data = segyio.collect(f.trace)
#         data = data[::4, :].reshape((240, 240, 2000))
#         np.savez(file_path + ".npz", data=data)
#     except Exception as e:
#         return f"Error opening file: {str(e)}"

@mcp.tool()
def read_seismic_info(file_path: str) -> str:
    """Get the seismic information from a SEG-Y file."""
    with segyio.open(file_path, "rb", ignore_geometry=True) as f:
        return segyio.tools.wrap(f.text[0])


@mcp.tool()
def read_seismic_binary_info(file_path: str) -> str:
    """Get the seismic binary header information from a SEG-Y file."""
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
    """Get the seismic trace header information from a SEG-Y file."""
    try:
        with segyio.open(file_path, "rb", ignore_geometry=True) as f:
            header = f.header[0]  # Read the first trace header
            return header
    except Exception as e:
        return f"Error reading header: {str(e)}"

@mcp.tool()
def butter_bandpass_filter(filepath:str, lowcut:int, highcut:int, shot:int, output_name:str)->str:
    r"""Apply Butterworth bandpass filter

    Apply Butterworth bandpass filter over time axis of input data

    Parameters
    ----------
    filepath : str
        Path to the npz file containing the data to be filtered
    data : np.ndarray
        1D or 2D array where filtering is applied along the last axis
    lowcut : int
        Lower cut-off frequency
    highcut : int
        Upper cut-off frequency
    output_name : str
        Short name of the output file to save the filtered data
    

    Returns
    -------
    y : np.ndarray
        Filtered data
    """
    fs = 1/4000
    data = np.load(filepath)["data"]
    order = 5
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass', analog=False)
    y = filtfilt(b, a, data, axis=-1)
    np.savez("result/"+ output_name + ".npz", data=y)
    return y

@mcp.tool()
def interpolate_gap(filepath:str,output_name:str)->str:
    """Interpolate gaps in the seismic data using sparsity promoting inversion."""
    nfft_x,nfft_t = 1024,1024

    data_miss = np.load(filepath)["data"]

    zero_rows = np.where(np.all(data_miss == 0, axis=1))[0]
    all_indices = np.arange(data_miss.shape[0])
    available_trace_indices = np.setdiff1d(all_indices, zero_rows)

    Rop = Restriction(dims=(data_miss.shape), iava=available_trace_indices, axis=0, dtype="float64")
    Fop = FFT2D(dims=(data_miss.shape), nffts=(nfft_x, nfft_t), dtype=np.complex128)

    F0op  = Rop * Fop.H
    data0 = np.delete(data_miss, zero_rows, axis=0).ravel()

    with pylops.disabled_ndarray_multiplication():
        pinv0, _, _ = fista(F0op, data0, niter=100, eps=200, 
                            eigsdict=dict(niter=5, tol=1e-2), show=True)
    data_recover = np.real(Fop.H*pinv0).reshape(data_miss.shape)
    np.savez("result/" + output_name + ".npz", data=data_recover)
    return data_recover

@mcp.tool()
def denoise(filepath:str,output_name:str)->str:
    """Denoise seismic data using TV Regularization.
    Args:
    filepath (str): Path to the npz file of the data.
    output name (str): Short name of the output file to save the denoised data.
    """
    data = np.load(filepath)["data"]
    image_denoise = denoise_tv_chambolle(data, weight=100000)
    np.savez("result/"+output_name+".npz",data=image_denoise)
    return image_denoise


if __name__ == "__main__":
    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=9000,
        log_level="debug",
        path="/atomic",
    )
