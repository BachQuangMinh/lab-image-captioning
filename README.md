# Lab: Image Captioning (simple)

This is a small lab to understand how to set up a core function of an application that creates a caption given an image.

Overview
- The example script is in `image_cap.py`.
- This project explores three practical extensions:
  1. A simple Gradio UI to demo the captioner to non‑technical stakeholders (`image_captioning_app.py`).
  2. Packaging the caption function so it can be imported and reused across files (`ImageCaptioning/image_captioning.py` or `image_cap_local.py`).
  3. Automating web scraping to caption images from Wikipedia (`automate_url_captioner.py`).

What I learned
- The processor is responsible for preprocessing input data so the model can understand it, and for decoding the model's output back into human-readable language.
- Organisations like Hugging Face provide pretrained models, enabling developers to reuse models rather than training from scratch.
- Use Gradio to build a simple UI for demos so non-technical stakeholders can interact with the model without running code.
- Package the caption function so it becomes a reusable component importable from other scripts and apps.
- Web scraping can be automated to find images (for example on Wikipedia) and feed them into the captioning pipeline for bulk or scheduled captioning.

Files
- `image_cap.py` — minimal example showing how to produce a caption for a local image.
- `image_cap_local.py` / `ImageCaptioning/image_captioning.py` — packaged caption function for import and reuse.
- `image_captioning_app.py` — example Gradio app to demo the model to non-technical users.
- `automate_url_captioner.py` — simple web scraping automation to download images from web pages (e.g., Wikipedia) and caption them.
- `requirements.txt` — Python dependencies.
- `test-image.png` — sample image.

Quick usage

1) Environment setup
- Create and activate a virtual environment:
  - python -m venv .venv
  - On macOS / Linux: source .venv/bin/activate
  - On Windows (PowerShell): .venv\Scripts\Activate.ps1
- Upgrade pip and install dependencies:
  - pip install --upgrade pip
  - pip install -r requirements.txt

2) Run the minimal example
- python image_cap.py

3) Run the Gradio demo (example)
- python image_captioning_app.py
- The script will print a local URL (http://127.0.0.1:7860) you can open in a browser to try the captioner.

Importing the caption function from other files
- Example: import the packaged function and call it from another script
```python
from ImageCaptioning.image_captioning import caption_image  # or from image_cap_local import caption_image

caption = caption_image("test-image.png")
print(caption)
```

Automating captions from Wikipedia (high level)
- `automate_url_captioner.py` contains a simple pipeline: fetch a Wikipedia page, find image URLs, download them, and call the caption function for each image.
- Use this for batch captioning or simple scraping demos. Respect robots.txt and site terms when scraping.

Device and local-run notes
- The base model expects reasonable compute (GPU recommended for speed).
- To run on CPU or choose device, add a device selection and move tensors/models:


```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)
```

If you lack a GPU, consider:
- Using a smaller model (if available).
- Running inference on CPU (slower).
- Using Hugging Face Inference API or a hosted endpoint to avoid local resource needs.

Notes
- This lab is for learning and experimentation.
- When using web scraping, follow site policies and rate limits.
