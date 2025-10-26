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

# Some ideas for extensions

Quick pointers to code in your workspace
- packaged caption function: `ImageCaptioning.image_captioning.caption_image` — image_captioning.py  
- local demo script: image_cap.py  
- local device-aware demo: image_cap_local.py  
- Gradio demo: image_captioning_app.py  
- Wikipedia scraper: automate_url_captioner.py  
- README and requirements: README.md, requirements.txt

Extension ideas (education + practical)
- Teach model internals and evaluation
  - Add an evaluation notebook that runs captions on a small labeled dataset and computes BLEU/METEOR/CIDEr so students see tradeoffs.
  - Add visualizations of attention or gradients (explainability).

- Improve API + reproducibility
  - Refactor the packaged captioner to accept file paths, PIL images, numpy arrays, and an explicit device parameter (CPU/GPU).
  - Expose a simple REST API (FastAPI) so other apps can call the captioner for experiments.

- Make experiments easier and faster
  - Add model caching, a small-checkpoint option, and inference tips (quantization / half precision) so CPU-only students can run it.
  - Add batching and async examples for throughput experiments.

- Data collection & active learning
  - Add a small labeling UI (Gradio) to let students correct captions and save corrected pairs to a CSV for fine-tuning.
  - Implement a simple active learning loop that collects the most uncertain samples for labeling.

- Dev hygiene & teaching materials
  - Add unit tests for the caption function, a dockerfile, and a CI workflow template.
  - Add a Jupyter notebook walkthrough that exercises the code and explains each step.

Starter changes

1) Make the packaged caption function more robust (accept path/PIL/ndarray, device selection, lazy model load). Replace the current module with this refactor:

````python
# ...existing code...
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch
from typing import Union, Optional

# ...existing code...
_processor = None
_model = None

def _ensure_model(device: Optional[str] = None):
    global _processor, _model
    if _processor is None or _model is None:
        _processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    if device:
        _model.to(device)

def caption_image(input_image: Union[str, np.ndarray, Image.Image], device: Optional[str] = None, max_length: int = 50) -> str:
    """
    Generate a caption for an image.
    Accepts a file path, a numpy array, or a PIL.Image.
    device: "cuda" or "cpu" (auto if None).
    """
    _ensure_model(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Normalize input to PIL Image
    if isinstance(input_image, str):
        pil_image = Image.open(input_image).convert("RGB")
    elif isinstance(input_image, np.ndarray):
        pil_image = Image.fromarray(input_image).convert("RGB")
    elif isinstance(input_image, Image.Image):
        pil_image = input_image.convert("RGB")
    else:
        raise TypeError("input_image must be a path, numpy array, or PIL.Image")

    text = "the image of"
    inputs = _processor(images=pil_image, text=text, return_tensors="pt")

    # Move inputs to device if model is on GPU
    model_device = next(_model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model.generate(**inputs, max_length=max_length)

    caption = _processor.decode(outputs[0], skip_special_tokens=True)
    return caption
# ...existing code...
````

- This keeps a single importable function `ImageCaptioning.image_captioning.caption_image` that is flexible for teaching experiments (file vs array vs PIL) and lets students toggle device.

2) Add a tiny REST endpoint to experiment with remote calls and reproducibility. Example FastAPI server:

````python
# ...existing code...
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
from ImageCaptioning.image_captioning import caption_image

app = FastAPI()

@app.post("/caption")
async def caption_endpoint(file: UploadFile = File(...), device: str = "cpu"):
    content = await file.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")
    caption = caption_image(pil, device=device)
    return {"filename": file.filename, "caption": caption}
# ...existing code...
````

Actionable next steps (pick 2–3 for a short course)
- Add an evaluation notebook + a tiny labeled dataset (10–50 images) to compute and discuss BLEU/CIDEr.
- Add a Gradio labeling app that saves corrected captions for a simple fine-tuning demo. Tie it to `ImageCaptioning.image_captioning.caption_image`.
- Add a CI job that runs a basic unit test asserting that captioning a sample image returns a non-empty string.