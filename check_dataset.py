from pathlib import Path
import cv2

DATASET_DIR = Path("dataset/DB1")

files = sorted(DATASET_DIR.glob("*.tif"))

print("Total TIFF files:", len(files))
print("-" * 40)

for file in files:
    image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(file.name, "FAILED TO LOAD")
        continue

    print(
        f"{file.name:20} "
        f"Size: {image.shape[1]} x {image.shape[0]}"
    )

print("-" * 40)
print("Dataset verification completed.")