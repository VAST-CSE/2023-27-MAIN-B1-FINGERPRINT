import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from preprocessing import preprocess_fingerprint

# Locate the original and processed fingerprint
original_dir = Path("dataset/DB1")
enhanced_dir = Path("results/enhanced")

files = sorted(original_dir.glob("*.tif"))

if not files:
    raise SystemExit("No original TIFF images found.")

image_path = files[0]
original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if original is None:
    raise SystemExit("Could not read original image.")

enhanced, binary, skeleton = preprocess_fingerprint(original)

images = [
    ("Original", original),
    ("Enhanced", enhanced),
    ("Binary", binary),
    ("Skeleton", skeleton)
]

plt.figure(figsize=(14, 10))

for i, (title, image) in enumerate(images, 1):
    plt.subplot(2, 2, i)
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")

plt.tight_layout()
plt.show()
