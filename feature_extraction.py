from pathlib import Path
import csv
import cv2
import numpy as np

from acquisition import load_fingerprints
from preprocessing import preprocess_fingerprint

OUTPUT_DIR = Path("results/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_minutiae(skeleton):
    """
    Detect candidate ridge endings and bifurcations
    using the crossing-number method.
    """
    # Convert white skeleton pixels to 1
    ridge = (skeleton > 0).astype(np.uint8)

    height, width = ridge.shape
    minutiae = []

    # Avoid the image border
    for y in range(1, height - 1):
        for x in range(1, width - 1):

            if ridge[y, x] == 0:
                continue

            # Eight neighbors in clockwise order
            neighbors = [
                ridge[y - 1, x],
                ridge[y - 1, x + 1],
                ridge[y, x + 1],
                ridge[y + 1, x + 1],
                ridge[y + 1, x],
                ridge[y + 1, x - 1],
                ridge[y, x - 1],
                ridge[y - 1, x - 1]
            ]

            # Crossing number
            transitions = sum(
                neighbors[i] != neighbors[(i + 1) % 8]
                for i in range(8)
            )

            crossing_number = transitions / 2

            if crossing_number == 1:
                kind = "ridge_ending"

            elif crossing_number == 3:
                kind = "bifurcation"

            else:
                continue

            minutiae.append({
                "x": x,
                "y": y,
                "type": kind
            })

    return minutiae


def main():
    images = load_fingerprints()

    if not images:
        print("No fingerprint images found.")
        return

    csv_path = OUTPUT_DIR / "minutiae.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["image", "x", "y", "type"]
        )
        writer.writeheader()

        for item in images:
            image = item["image"]
            filename = item["filename"]

            # Preprocess the original image
            _, _, skeleton = preprocess_fingerprint(image)

            # Extract candidate minutiae
            points = extract_minutiae(skeleton)

            # Visualize the detections
            visualization = cv2.cvtColor(
                image, cv2.COLOR_GRAY2BGR
            )

            for point in points:
                x = point["x"]
                y = point["y"]

                # Different marker shapes are not needed;
                # colors distinguish the two feature types.
                if point["type"] == "ridge_ending":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.circle(
                    visualization,
                    (x, y),
                    3,
                    color,
                    -1
                )

                writer.writerow({
                    "image": filename,
                    **point
                })

            output_path = OUTPUT_DIR / (
                Path(filename).stem + "_minutiae.png"
            )

            cv2.imwrite(
                str(output_path),
                visualization
            )

            print(
                f"{filename}: "
                f"{len(points)} candidate minutiae"
            )

    print("\nFeature extraction completed.")
    print("CSV saved to:", csv_path.resolve())
    print("Visualizations saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
