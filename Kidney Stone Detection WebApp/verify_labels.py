"""
Randomly samples N images + their YOLO-format labels, draws the bounding boxes,
and saves them to a folder so you can eyeball whether the auto-generated labels
actually line up with the stones in the CT scans.

Just edit the three paths below and run: python verify_labels.py
"""

import os
import random
from pathlib import Path

import cv2

# --- EDIT THESE THREE PATHS ---
IMAGES_DIR = r'F:\dataset\Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection A Resource for Deep Learning Research\Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection A Resource for Deep Learning Research\Kindy Stone Dataset\Augmented Dataset\Stone'
LABELS_DIR = r'F:\Kidney Stone\runs_v26x\content\runs\detect\train\weights\runs\detect\output_labels\results\labels'
OUTPUT_DIR = r'F:\Kidney Stone\label_check_samples'
# --------------------------------

NUM_SAMPLES = 30  # how many random images to check

os.makedirs(OUTPUT_DIR, exist_ok=True)

label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
print(f"Total labels found: {len(label_files)}")

sample = random.sample(label_files, min(NUM_SAMPLES, len(label_files)))

checked = 0
for label_file in sample:
    stem = Path(label_file).stem

    # find the matching image (try common extensions)
    img_path = None
    for ext in ['.jpg', '.jpeg', '.png']:
        candidate = os.path.join(IMAGES_DIR, stem + ext)
        if os.path.isfile(candidate):
            img_path = candidate
            break

    if img_path is None:
        print(f"⚠️  No matching image found for label: {label_file}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️  Could not read image: {img_path}")
        continue

    h, w = img.shape[:2]

    with open(os.path.join(LABELS_DIR, label_file), 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, xc, yc, bw, bh = parts[:5]
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "stone", (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out_path = os.path.join(OUTPUT_DIR, f"check_{stem}.jpg")
    cv2.imwrite(out_path, img)
    checked += 1

print(f"\n✅ Done. Saved {checked} labeled preview images to:\n{OUTPUT_DIR}")
print("Open that folder and look through the images — do the green boxes")
print("actually sit on top of the kidney stones?")