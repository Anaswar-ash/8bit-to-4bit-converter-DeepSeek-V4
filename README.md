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

The primary conversion script is `convert_fp8_to_int4.py`. It requires the HuggingFace URL to the repository and an output directory.

### Basic Conversion
To convert a full model:
```bash
python convert_fp8_to_int4.py --hf_url deepseek-ai/DeepSeek-V4-Flash --out_dir ./v4_int4
```

### Advanced: Surgical Indexer Download
DeepSeek-V4 has massive ~750 GB repositories. If you only want to download and extract the Lightning Indexer tensors (which are critical for sparse attention) and skip the rest of the model, you can use the `--indexer` flag. 

This flag uses **HTTP Range requests** to surgically extract only the bytes needed for the `idx_wq_b` and `idx_wproj` matrices without downloading the entire FP8 shard files.

```bash
python convert_fp8_to_int4.py --hf_url deepseek-ai/DeepSeek-V4-Flash --out_dir ./v4_int4 --indexer
```
*(This will generate `out-idx-XXXXX.safetensors` files containing the INT8 compressed indexer matrices in just a few minutes, totaling ~5 GB).*
