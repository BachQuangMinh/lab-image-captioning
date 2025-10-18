from ImageCaptioning.image_captioning import caption_image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO
from PIL import Image
import numpy as np

# URL of the page to scrape
url = "https://en.wikipedia.org/wiki/IBM"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 1) Fetch page
resp = requests.get(url, headers=headers, timeout=20)
resp.raise_for_status()

# 2) Parse HTML
soup = BeautifulSoup(resp.text, 'html.parser')

# 3) Find images
img_elements = soup.find_all('img')

def to_absolute_image_url(page_url: str, src: str) -> str | None:
    if not src:
        return None
    # protocol-relative (e.g., //upload.wikimedia.org/...)
    if src.startswith('//'):
        return 'https:' + src
    # absolute (http/https)
    if src.startswith('http://') or src.startswith('https://'):
        return src
    # relative (/static/... or images/...)
    return urljoin(page_url, src)

# 4) Iterate and caption
for img_el in img_elements:
    # try both common attributes
    src = img_el.get('src') or img_el.get('data-src') or img_el.get('data-original')
    img_url = to_absolute_image_url(url, src)
    if not img_url:
        continue

    # Skip non-photographic assets and tiny icons
    lower = img_url.lower()
    if any(ext in lower for ext in ('.svg', '.gif')):
        continue

    try:
        r = requests.get(img_url, headers=headers, timeout=20)
        r.raise_for_status()

        # Decode to PIL image then to numpy array (what caption_image expects)
        pil_img = Image.open(BytesIO(r.content)).convert('RGB')
        # Optional: skip tiny images (logos/icons)
        if min(pil_img.size) < 64:  # width/height threshold
            continue

        arr = np.array(pil_img)
        caption = caption_image(arr)  # caption_image(Image.fromarray(...)) will now work
        print(f"URL: {img_url}")
        print("Caption:", caption)

    except Exception as e:
        # Don’t crash the loop on one bad image
        print(f"Skipping {img_url}: {e}")