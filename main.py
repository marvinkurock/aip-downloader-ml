from io import BytesIO
import sys
import base64
import re
import json
import requests
from urllib.parse import urljoin
import tensorflow as tf
import numpy as np
from PIL import Image

model = None
class_names = ['AD', 'GND', 'VAD']  # Your class names

def print_progress(percent):
    percent = round(percent)
    bar = "#" * (percent // 2)  # 50 chars wide
    print(f"\r[{bar:<50}] {percent}%", end="")

baseurl = "https://aip.dfs.de/BasicVFR/pages/"
debug = False

def predict(bytes: bytes):
  print("predicting")
  image = Image.open(BytesIO(bytes))
  image = image.resize((256, 256))
  img_array = np.array(image)

  # Check if the image has 3 channels (RGB)
  if img_array.shape[-1] != 3:
      raise ValueError("The image must have 3 color channels (RGB)")

  # Normalize the image (ensure it's the same as during training)
  img_array = img_array / 255.0

  # Add batch dimension (shape becomes [1, 224, 224, 3])
  img_array = np.expand_dims(img_array, axis=0)

  p = model.predict(img_array)
  predicted_class = class_names[np.argmax(p)]
  return predicted_class

def get_airports():
  result = []
  with open('./airports.json') as f:
    data = json.load(f)
    for a in data:
      result.append({ "name": a["n"], "code": a["m"] })
  return result


def download():
  airports = get_airports()
  for c, a in enumerate(airports):
    type_counter = {
      "AD": 0,
      "GND": 0,
      "VAD": 0
    }
    print_progress(c / len(airports) * 100)
    code_match = re.findall(r'\s([A-Z]{4})', a['name'])
    if len(code_match) == 0:
      print("no icao code found. skipping")
      continue
    code = code_match[0]
    c = requests.get(f'https://aip.dfs.de/BasicVFR/pages/{a["code"]}.html')
    match = re.search(r'url=([^"]+)', c.text)
    if match:
      url = match.group(1)
      ap_url = urljoin(baseurl, url)
      c2 = requests.get(ap_url)
      hrefs = re.findall(r'href=["\'](\.\./pages/[^"\']*)["\']', c2.text)
      for i, u in enumerate(hrefs):
        cu = urljoin(ap_url, u)

        c3 = requests.get(cu)
        html = c3.text
        match = re.search(r'<img[^>]*id=["\']imgAIP["\'][^>]*src=["\']([^"\']+)["\']', html)
        if match:
          src = match.group(1)
          b64 = src.replace("data:image/png;base64,", "")
          binary = base64.b64decode(b64)
          t = predict(binary)
          print(t)
          suffix = "" if type_counter[t] == 0 else type_counter[t] 
          with open(f"charts/{code}_airport_{t}{suffix}.png", "wb") as f:
            f.write(binary)
            type_counter[t] = type_counter[t] + 1
        else:
          print("Image with id 'imgAIP' not found.")
    else:
      print("URL not found.")
      continue


def main():
  args = sys.argv
  print(args)
  if len(args) < 2:
    print("this is main")
    exit(0)

  if args[1] == "download":
    global model 
    model = tf.keras.models.load_model('./model-1.0.keras')
    download()

if __name__ == "__main__":
  main()

