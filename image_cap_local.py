import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import torch

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Suggested: select device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load your image, DON'T FORGET TO WRITE YOUR IMAGE NAME
img_path = "test-image.png"
# convert it into an RGB format 
image = Image.open(img_path).convert('RGB')

# You do not need a question for image captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt")

# Move inputs to the selected device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate a caption for the image (use no_grad for inference)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print(caption)