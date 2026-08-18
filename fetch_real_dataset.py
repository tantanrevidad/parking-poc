"""
fetch_real_dataset.py
----------------------
Downloads a small, curated subset of the openalpr/benchmarks "endtoend"
dataset — real photographs of parked/driving vehicles with license plates,
released publicly for ALPR (Automatic License Plate Recognition) research
and benchmarking, with hand-verified ground-truth plate text + bounding box
for each image.

Source:  https://github.com/openalpr/benchmarks  (AGPLv3)
Cite:    OpenALPR, "OpenALPR Benchmark Dataset", 2016.
         https://github.com/openalpr/benchmarks/tree/master/endtoend

WHY THIS DATASET, SPECIFICALLY:
  - It's an established, widely-cited academic/research ALPR benchmark
    (used in NVIDIA's own ALPR tutorials and multiple peer-reviewed papers)
    — not photos scraped ad hoc from random public sources.
  - Every image ships with a hand-verified ground-truth plate string, which
    is what lets us compute REAL OCR accuracy (exact-match rate,
    character-level accuracy) instead of just "it produced some text."

PRIVACY / LICENSING NOTE:
  These are real photographs containing real (though several years old,
  publicly benchmark-released) license plates. Per this project's own
  design principles (see the original implementation plan's Section 10),
  this script deliberately does NOT commit these images to source control
  or bundle them into any distributed package — cv-demo/source-frames/ is
  gitignored. Running this script re-downloads fresh from the authoritative
  source each time, which is the standard, correct way to depend on a
  third-party research dataset. The repo is AGPLv3-licensed; this script
  only downloads and locally evaluates against the data for feasibility
  testing and does not redistribute it.

Usage:
    python fetch_real_dataset.py
"""

import os
import urllib.request
import json

RAW_BASE = "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend"

# Curated subset: real US + EU plates, moderate image sizes, a mix of plate
# lengths and characters (including several with confusable characters like
# 0/O, so the matcher has something real to prove itself against).
CURATED_FILES = [
    ("us", "0b86cecf-67d1-4fc0-87c9-b36b0ee228bb"),
    ("us", "12c6cb72-3ea3-49e7-b381-e0cdfc5e8960"),
    ("us", "1e241dc8-8f18-4955-8988-03a0ab49f813"),
    ("us", "21d8c31d-3deb-494b-9c63-c0223306fd82"),
    ("us", "22e54a62-57a8-4a0a-88c1-4b9758f67651"),
    ("us", "37170dd1-2802-4e38-b982-c5d07c64ff67"),
    ("us", "3850ba91-3c64-4c64-acba-0c46b61ec0da"),
    ("us", "car1"),
    ("us", "car13"),
    ("us", "car19"),
    ("eu", "eu1"),
    ("eu", "eu3"),
    ("eu", "eu5"),
    ("eu", "eu6"),
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv-demo", "source-frames")


def parse_groundtruth_line(text):
    """Format: <filename>\\t<x>\\t<y>\\t<w>\\t<h>\\t<PLATE_TEXT>"""
    parts = text.strip().split("\t")
    if len(parts) != 6:
        parts = text.strip().split()
    filename, x, y, w, h, plate = parts
    return {
        "filename": filename,
        "bbox": [int(x), int(y), int(w), int(h)],
        "plate": plate,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []

    print(f"Fetching {len(CURATED_FILES)} real images from openalpr/benchmarks ...")
    for subset, stem in CURATED_FILES:
        jpg_url = f"{RAW_BASE}/{subset}/{stem}.jpg"
        txt_url = f"{RAW_BASE}/{subset}/{stem}.txt"
        jpg_path = os.path.join(OUTPUT_DIR, f"{subset}_{stem}.jpg")
        txt_path = os.path.join(OUTPUT_DIR, f"{subset}_{stem}.txt")

        try:
            urllib.request.urlretrieve(jpg_url, jpg_path)
            urllib.request.urlretrieve(txt_url, txt_path)
        except Exception as e:
            print(f"  FAILED {subset}/{stem}: {e}")
            continue

        with open(txt_path) as f:
            gt = parse_groundtruth_line(f.read())
        gt["image_file"] = f"{subset}_{stem}.jpg"
        gt["subset"] = subset
        manifest.append(gt)
        print(f"  OK  {subset}/{stem}  ->  plate: {gt['plate']}")

    manifest_path = os.path.join(OUTPUT_DIR, "..", "ground_truth.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDownloaded {len(manifest)}/{len(CURATED_FILES)} images.")
    print(f"Ground truth manifest: {manifest_path}")
    print("\nSource: https://github.com/openalpr/benchmarks (AGPLv3)")
    print("These images are NOT committed to this repo — re-run this script to refresh them.")


if __name__ == "__main__":
    main()
