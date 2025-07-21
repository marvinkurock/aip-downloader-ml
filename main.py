#! /usr/bin/env python
import zipfile
from io import BytesIO
from datetime import datetime
import os
import sys
import base64
import re
import json
import requests
from urllib.parse import urljoin
import tensorflow as tf
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)
logging.basicConfig()

get_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36"
  }

model = None
class_names = ['AD', 'GND', 'VAD']  # Your class names
chart_types = {
  "AD": "airport",
  "VAD": "arrival",
  "GND": "taxi"
}

def print_progress(percent):
    percent = round(percent)
    bar = "#" * (percent // 2)  # 50 chars wide
    print(f"\r[{bar:<50}] {percent}%", end="")

baseurl = "https://aip.dfs.de/BasicVFR/"
debug = False

def predict(image):
  print("predicting")
  resized = image.resize((256, 256))
  img_array = np.array(resized)

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
    aname = a['name']
    if len(code_match) == 0:
      logger.debug(f"no icao code found for '{aname}'. skipping")
      continue
    code = code_match[0]
    url = f'{baseurl}pages/{a["code"]}.html'
    c = requests.get(url, headers=get_headers)
    match = re.search(r'url=([^"]+)', c.text)

    if match:
      url = match.group(1)
      ap_url = urljoin(baseurl + "pages/", url)
      c2 = requests.get(ap_url, headers=get_headers)
      logger.debug(f"status chart page: {ap_url}, {c2.status_code}")
      hrefs = re.findall(r'href=["\'](\.\./pages/[^"\']*)["\']', c2.text)
      for u in hrefs:
        cu = urljoin(ap_url, u)

        c3 = requests.get(cu, headers=get_headers)
        logger.debug(f"img url: {cu}")
        html = c3.text
        match = re.search(r'<img[^>]*id=["\']imgAIP["\'][^>]*src=["\']([^"\']+)["\']', html)
        if match:
          src = match.group(1)
          b64 = src.replace("data:image/png;base64,", "")
          binary = base64.b64decode(b64)
          img = Image.open(BytesIO(binary))
          t = predict(img)
          logger.debug(t)
          type_counter[t] = type_counter[t] + 1
          suffix = "" if type_counter[t] == 1 else type_counter[t] 
          out_name = f"charts/byop/{code}_{chart_types[t]}_{t}{suffix}.pdf"
          img.save(out_name)

        else:
          logger.debug("Image with id 'imgAIP' not found.")
    else:
      logger.debug("URL not found.")
      continue

def update_manifest():
  tpl = None
  with open('manifest.json') as f:
    tpl = json.load(f)
  v = tpl['version']
  v = round(v+0.1, 1)
  tpl['version'] = v
  now = datetime.now().strftime('%Y%m%dT%H:%M:%SZ')
  tpl['effectiveDate'] = now
  print(tpl)
  with open('charts/manifest.json', 'w') as f:
    json.dump(tpl, f)
  with open('manifest.json', 'w') as f:
    json.dump(tpl, f)

  return tpl
def create_zip(manifest):
  f_name = f'mk_aip_bundle_{manifest["version"]}.zip'
  with zipfile.ZipFile(f_name, 'w') as zf:
    for root, _, files in os.walk('charts'):
      for file in files:
        fp = os.path.join(root, file)
        arcname = fp
        zf.write(fp, arcname)

def main():
  args = sys.argv
  if len(args) < 2:
    print("this is main")
    exit(0)

  logLevel = logging.INFO

  if len(args) > 2:
    if args[2] == "-d":
      logLevel = logging.DEBUG
      print("setting logleel debug")
  logger.setLevel(logLevel)
  logger.debug(args)
  if args[1] == 'debug':
    m = update_manifest()
    create_zip(m)

  if args[1] == "new":
    global model 
    files = [f for f in os.listdir('charts/byop/')]
    for f in files:
      os.remove('charts/byop/'+f)
    try:
      os.remove('charts/manifest.json')
    except:
      pass
    model = tf.keras.models.load_model('./model-1.0.keras')
    download()
    m = update_manifest()
    create_zip(m)


if __name__ == "__main__":
  main()

