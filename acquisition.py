from pathlib import Path
import cv2

DATASET_DIR = Path("dataset/DB1")


def load_fingerprints():
    """Load fingerprint images from the dataset folder."""

    if not DATASET_DIR.exists():
        print(f"Dataset folder not found: {DATASET_DIR.resolve()}")
        return []

    extensions = ["*.tif", "*.tiff", "*.png", "*.bmp"]
    image_paths = []

    for extension in extensions:
        image_paths.extend(DATASET_DIR.glob(extension))

    image_paths = sorted(image_paths)
    fingerprints = []

    for path in image_paths:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f"Could not read: {path.name}")
            continue

        fingerprints.append({
            "filename": path.name,
            "image": image
        })

    print(f"Successfully loaded {len(fingerprints)} fingerprints.")

    return fingerprints


if __name__ == "__main__":
    images = load_fingerprints()
    print("Acquisition test completed.")
