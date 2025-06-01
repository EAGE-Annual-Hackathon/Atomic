import tempfile

import matplotlib.pyplot as plt
from fastmcp import FastMCP, Image
import segyio
#from pydantic_ai import Agent
from segyio.tools import cube
from starlette.requests import Request
from starlette.responses import JSONResponse
import numpy as np

mcp = FastMCP(
    name="Atomic",
    instructions="""
        This server provides an automatic workflow of seismic processing from data loading, visualization, processing and presenting!
        """,
)

@mcp.tool()
def open_seismic_data(file_path: str) -> str:
    """Always use this function to open a SEG-Y file at the beginning of the chat. To make the data available for other functions."""
    try:
        f = segyio.open(file_path, "rb", ignore_geometry=True)
        data = segyio.collect(f.trace)
        data = data[::4,:].reshape((240,240,2000))
        np.savez(file_path + ".npz", data=data)
    except Exception as e:
        return f"Error opening file: {str(e)}"
    
@mcp.tool()
def read_seismic_info(file_path:str) -> str:
    """Read the seismic information from a SEG-Y file."""
    with segyio.open(file_path, "rb", ignore_geometry=True) as f:
        return segyio.tools.wrap(f.text[0])

@mcp.tool()
def read_seismic_binary_info(file_path: str) -> str:
    """Read seismic binary header information from a SEG-Y file."""
    try:
        with segyio.open(file_path, "rb",ignore_geometry=True) as f:
            header = f.bin  # Read the first trace header
            return header
    except Exception as e:
        return f"Error reading header: {str(e)}"

@mcp.tool()
def read_trace_header(file_path: str,) -> str:
    """Read the seismic trace header information from a SEG-Y file."""
    try:
        with segyio.open(file_path, "rb", ignore_geometry=True) as f:
            header = f.header[0]  # Read the first trace header
            return header
    except Exception as e:
        return f"Error reading header: {str(e)}"

@mcp.tool()
def visualize_2d(filepath:str,shot:int,title:str)-> Image:
    """Visualize a 2D seismic section. 
    
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
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        plt.figure(figsize=(8, 6))
        plt.imshow(data[:,shot,:].T, aspect="auto", cmap="gray",vmin=-1e-8,vmax=1e-8)
        # Title uses the correct filename of the SEG-Y file
        plt.title("Data shot %d"%shot)
        plt.xlabel("Xline")
        plt.ylabel("Samples")
        plt.colorbar(label="Amplitude")
        plt.tight_layout()
        plt.savefig(tmpfile.name, format="jpeg")
        plt.close()
        jpeg_path = tmpfile.name

    return Image(path=jpeg_path, format="jpeg")

if __name__ == "__main__":
    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=9000,
        log_level="debug",
        path="/hello_world_mcp",
    )
