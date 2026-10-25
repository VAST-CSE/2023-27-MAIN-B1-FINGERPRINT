from pathlib import Path
import cv2
import numpy as np
from skimage.morphology import skeletonize


# ==========================================
# 1. CONFIGURATION
# ==========================================

INPUT_DIR = Path("dataset/DB1")
OUTPUT_DIR = Path("results")

ENHANCED_DIR = OUTPUT_DIR / "enhanced"
BINARY_DIR = OUTPUT_DIR / "binary"
SKELETON_DIR = OUTPUT_DIR / "skeleton"


# ==========================================
# 2. CREATE OUTPUT FOLDERS
# ==========================================

for folder in [ENHANCED_DIR, BINARY_DIR, SKELETON_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ==========================================
# 3. FINGERPRINT PREPROCESSING
# ==========================================

def preprocess_fingerprint(image):

    # Step 1: Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image)

    # Step 2: Reduce minor noise
    denoised = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    # Step 3: Convert fingerprint to binary
    # White pixels represent foreground ridges.
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    # Step 4: Remove small connected components
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 10:
            cleaned[labels == i] = 255

    # Step 5: Skeletonization
    skeleton = skeletonize(cleaned > 0)

    skeleton_image = skeleton.astype(np.uint8) * 255

    return enhanced, cleaned, skeleton_image


# ==========================================
# 4. LOAD DATASET WITHOUT DUPLICATE MATCHES
# ==========================================

def get_fingerprint_files():

    if not INPUT_DIR.exists():
        print("ERROR: Dataset folder not found:")
        print(INPUT_DIR.resolve())
        return []

    valid_extensions = {".tif", ".tiff"}

    # Use one directory scan to avoid duplicate matches.
    files = sorted(
        path
        for path in INPUT_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
    )

    return files


# ==========================================
# 5. PROCESS ALL FINGERPRINTS
# ==========================================

def main():

    print("Starting fingerprint preprocessing...")

    image_paths = get_fingerprint_files()

    if not image_paths:
        print("No TIFF fingerprint images found.")
        print("Check INPUT_DIR:", INPUT_DIR.resolve())
        return

    print("Unique image files found:", len(image_paths))

    successful = 0
    failed = 0

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print("Failed to read:", image_path.name)
            failed += 1
            continue

        enhanced, binary, skeleton = preprocess_fingerprint(image)

        # Save enhanced image
        ok1 = cv2.imwrite(
            str(ENHANCED_DIR / image_path.name),
            enhanced
        )

        # Save binary image
        ok2 = cv2.imwrite(
            str(BINARY_DIR / image_path.name),
            binary
        )

        # Save skeleton image
        ok3 = cv2.imwrite(
            str(SKELETON_DIR / image_path.name),
            skeleton
        )

        if not all([ok1, ok2, ok3]):
            print("Failed to save results:", image_path.name)
            failed += 1
            continue

        successful += 1
        print(f"Processed: {image_path.name}")

    print("\n" + "=" * 40)
    print("PREPROCESSING SUMMARY")
    print("=" * 40)
    print("Input images found:", len(image_paths))
    print("Successfully processed:", successful)
    print("Failed:", failed)
    print("Enhanced folder:", ENHANCED_DIR.resolve())
    print("Binary folder:", BINARY_DIR.resolve())
    print("Skeleton folder:", SKELETON_DIR.resolve())
    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
