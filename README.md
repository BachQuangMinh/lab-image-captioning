# Lab: Image Captioning (simple)

This is a small lab to understand how to set up a core function of an application that creates a caption given an image.

Overview
- The example code is in [image_cap.py](image_captioning_project/image_cap.py).
- The script demonstrates usage of a processor and model to convert an image into a human-readable caption:
  - processor: [`image_cap.processor`](image_captioning_project/image_cap.py)
  - model: [`image_cap.model`](image_captioning_project/image_cap.py)

What I learned
- The processor is responsible for preprocessing input data so the model can understand it, and for decoding the model's output back into human-readable language.
- Organisations like Hugging Face provide pretrained models (so developers do not always need to train from scratch), which makes AI more accessible.

Files
- [image_cap.py](image_captioning_project/image_cap.py) — main example script
- [requirements.txt](image_captioning_project/requirements.txt) — Python dependencies
- [test-image.png](image_captioning_project/test-image.png) — sample image used by the script

How to run this lab (environment setup)
1. Create and activate a virtual environment:
   - Python venv:
     - python -m venv .venv
     - On macOS / Linux: source .venv/bin/activate
     - On Windows (PowerShell): .venv\Scripts\Activate.ps1
2. Upgrade pip and install dependencies:
   - pip install --upgrade pip
   - pip install -r image_captioning_project/requirements.txt
3. Run the example:
   - python image_captioning_project/image_cap.py

Notes about hardware and running locally
- This lab expects a machine with sufficient GPU capacity for reasonable performance when using the default model.
- If you do not have a GPU or you want to run on a local CPU-only machine, consider the following modifications:
  - Use a smaller model or run the model on CPU (will be slower).
  - Move tensors and model onto the selected device in the script (example modification below).
  - Alternatively, use Hugging Face Inference API or a hosted inference endpoint to avoid local GPU requirements.

Suggested modifications for CPU / device selection
- See the suggested changes in the repository to set the device and move inputs to the device. Example modifications are shown in [image_cap.py](image_captioning_project/image_cap.py).

License / Notes
- This lab is for learning/experimentation only.
