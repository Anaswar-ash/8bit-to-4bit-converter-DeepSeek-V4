# 8bit-to-4bit Converter

A standalone Python tool designed to drastically compress specific Large Language Models (like **DeepSeek-V4-Flash** and **GLM-5**) from HuggingFace.

This script fetches massive `FP8` (8-bit) neural network weights directly from the HuggingFace Hub, converts them to highly compressed `INT4` (4-bit) safetensors tailored for the Colibri/DeepSeek-C engines, and then immediately deletes the heavy FP8 files to save disk space.

*Note: This tool uses custom tensor matching logic and a proprietary INT4 packing format. It is not designed for standard PyTorch/Transformers AWQ/GPTQ formats, nor is it universally compatible with models like Llama-3.*

## Features
- **Disk-Safe Streaming:** Downloads, converts, and deletes one shard at a time. It will never fill up your hard drive with the full uncompressed model!
- **Multi-threaded Downloading:** Uses multiple parallel streams to maximize your network bandwidth.
- **Resumable:** If your internet drops, simply run the command again. It will resume exactly where it left off without losing progress.

## Installation

1. Install Python (3.10+ recommended).
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the `convert_dsv4.py` script and pass it the name of the HuggingFace repository and your desired output directory.

**Example for DeepSeek-V4:**
```bash
python convert_dsv4.py --repo deepseek-ai/DeepSeek-V4-Flash-Base --outdir ./v4_int4
```

*(This will slowly generate the `v4_int4` folder containing the highly compressed model ready for C-engines!)*
