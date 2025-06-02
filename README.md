# Automic

This repository provides reproducible material for Automatic Seismic Processing project for EAGE Hackathon 2025

## Project Description
In this project we try to automatize the quality control of the seismic dataset using Volve Field Data using Agentic AI. The Agent will try to see the data, evaluate, decide what processing necessary and do the processing. The process showcase here are interpolating seismic gaps, denoising, and frequency filtering. But it can also extended to others processing steps.

## Project Structure
The repository is organized as follows:

* :open_file_folder: **data**: A folder containing the data for test
* :open_file_folder: **data**: A folder containing the data or instructions on how to obtain it.
* :open_file_folder: **notebooks**: Jupyter notebooks that document the application of IntraSesimic to the inversion of the synthetic Marmousi data.


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

