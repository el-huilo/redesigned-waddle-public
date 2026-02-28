# AI Image Generation App

`Public version of the repo`

**Gradio web interface** for AI image generation using Stable Diffusion and related models. Originally built as a personal tool, later expanded with GIF generation and model management. Runs in Google Colab or locally.

## What it does

The app provides a unified interface for generating images and animations with various diffusion models. Users can:

- **Text-to-Image** – generate images from a prompt (SDXL or SD models).
- **Image-to-Image** – transform an existing image using a prompt.
- **AnimateDiff (GIF generation)** – create short animations from text prompts (SD models only).
- **Multi‑frame prompting** – assign different prompts to different frames (via FreeNoise) for longer, controlled GIFs.
- **Model management** – load models from local `.safetensors` files or download them directly from a URL (e.g. CivitAI).

## Technologies

- **Python 3**
- **PyTorch**
- **Gradio** – web UI framework.
- **Diffusers** (Hugging Face) – pipeline management for Stable Diffusion, AnimateDiff, HunyuanVideo.
- **Google Colab integration** – install script, cloudflared tunnel for public access.
- **Cloudflared** – exposes local Gradio server via a TryCloudflare URL.

## Project history

- **2024** – started as a simple Colab notebook for personal image generation.
- **First month** – active development.
- **Later** – used as‑is.
- **2026** – repository archived, now open‑sourced for portfolio purposes.

## How to run

### In Google Colab (recommended)

1. Open the provided notebook (`Untitled.ipynb`) in Colab with a GPU runtime.
2. Run the first cell. It will:
   - Install dependencies.
   - Set up cloudflared.
   - Clone the repository.
   - Start `manag.py`.
3. After a few seconds, the logs will show a cloudflared URL (e.g. `https://something.trycloudflare.com`). Open it in your browser.
4. (Optional) Append `/?__theme=light` or `/?__theme=dark` to the URL to switch themes.

### Locally

1. Clone the repository.
2. Install dependencies: `pip install gradio diffusers torch xformers invisible_watermark ...` (see notebook for full list).
3. Run `python app.py`. The interface will be available at `http://127.0.0.1:7860`.

## Structure

- `app.py` – main Gradio application; contains UI layout, generation logic, model loading.
- `manag.py` – launcher script that starts cloudflared tunnel and the Gradio app (used in Colab).
- `Untitled.ipynb` – Colab notebook that sets up the environment and runs `manag.py`.

## Notable implementation details

- **Dynamic model loading** – models are loaded from single `.safetensors` files using Diffusers’ `from_single_file`. The type (SDXL/SD/HunyuanVideo) is selected by the user.
- **AnimateDiff integration** – for SD models, an AnimateDiff pipeline is built from the base pipeline. FreeNoise is enabled to allow longer GIFs (up to 196 frames) and per‑frame prompts.
- **Prompt dictionary** – users can store multiple prompts keyed by frame number; these are fed to AnimateDiff for multi‑frame control.
- **Device detection** – automatically uses CUDA if available, falls back to CPU.
- **Version check** – queries GitHub releases to compare with the current version (`v3.7` as of latest).
- **Image storage** – generated images are kept in memory and displayed in a gallery; users can delete individual items; GIFs are exported to disk; gallery allows to download and upload content.

## Status

The project is **complete but discontinued**. All core features (SD/SDXL text‑to‑image, image‑to‑image, AnimateDiff GIFs) work as expected.  
Known limitations:

- HunyuanVideo pipeline is disabled (out‑of‑memory on T4 Colab GPUs).
- Maximum image size is limited to 1024×1024 (to fit T4 memory).
