# Automic

This repository contains reproducible materials for the **Automatic Seismic Processing** project developed for the **EAGE Hackathon 2025**.

## 📌 Project Description

In this project, we aim to automate the quality control of seismic datasets using **Volve Field Data** and **Agentic AI**. The agent is designed to:
- Analyze the data  
- Evaluate its quality  
- Decide on necessary processing steps  
- Execute the processing

The demonstrated processes include:
- Interpolating seismic data gaps  
- Denoising  
- Frequency filtering  

The system can also be extended to support additional processing steps. For more information, refer to the provided PowerPoint presentation in the repository.


## Project Structure
The repository is organized as follows:

* :open_file_folder: **data**: A folder containing the data for test, processing results, and figures
* :open_file_folder: **dataref**: A folder containing the data or instructions on how to obtain it.

## Getting Started :space_invader: :robot:
Please install Microsoft Visual Studio Code Insiders (we need this for the latest MCP features):
- https://code.visualstudio.com/insiders/

Make sure to install the `uv` package manager (think Anaconda but much faster): 
- https://docs.astral.sh/uv/#installation

After installation start the server with the following: 
```bash
uv run hello_world_mcp/server.py
```
Navigate to `.vscode/mcp.json` once the server is running and click on `Start` for the HelloWorldMCP to start the MCP server. 

## Interacting with the MCP Server

### Github Copilot
Go to your Github Copilot window, activate `Ask`-mode and add the tool into your context, or when in `Agent`-mode add the tool through the MCP configuration in the lower left corner. 

