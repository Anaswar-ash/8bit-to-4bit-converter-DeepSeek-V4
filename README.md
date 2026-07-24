# DeepSeek-V4-Flash 8-bit to 4-bit Converter

A specialized, highly-efficient tool to download and quantize the massive **DeepSeek-V4-Flash** (284B MoE) model from Hugging Face into a memory-mapped INT4 format.

This converter is explicitly designed to prepare weights for the [DeepSeek-C Engine](https://github.com/Anaswar-ash/DeepSeek-C).

## Features
- **Disk-Safe Pipeline:** Downloads massive FP8 shards from Hugging Face, converts them to INT4, and *immediately deletes the original FP8 shard*. This prevents your hard drive from filling up (the raw model is over 700 GB, but you only need ~160 GB of free space to run this tool).
- **MTP Speculative Decoding:** Native support for isolating and downloading the Multi-Token Prediction (MTP) drafting heads. The MTP heads are kept at INT8 precision to ensure high draft-acceptance rates for 2x generation speeds.
- **Resumable Downloads:** If your network drops or you stop the script, running the exact same command will seamlessly resume the download exactly where it left off.

## Prerequisites
- **Python 3.9+**
- **Storage:** At least 165 GB of free NVMe SSD space.
- **Dependencies:**
  ```powershell
  pip install torch safetensors huggingface_hub numpy
  ```

## Usage

### 1. Convert Main Model Weights
Run the script to begin downloading and quantizing the core model. You can specify any output directory.

```powershell
python convert_dsv4.py --repo deepseek-ai/DeepSeek-V4-Flash --outdir ./v4_int4
```

### 2. Extract MTP Heads
Once the main model finishes, run the script a second time using the `--mtp` flag. This will locate the specific layer containing the Multi-Token Prediction draft heads (Layer 43) and save them to your folder.

```powershell
python convert_dsv4.py --repo deepseek-ai/DeepSeek-V4-Flash --outdir ./v4_int4 --mtp
```

### 3. Run the Engine
Once both commands finish, your folder will contain all the necessary `out-*.safetensors`, `out-mtp-*.safetensors`, `out-idx-*.safetensors`, `config.json`, and `tokenizer.json` files.

You can instantly boot the DeepSeek-C engine by providing the path:
```powershell
./dsv4.exe ./v4_int4
```
