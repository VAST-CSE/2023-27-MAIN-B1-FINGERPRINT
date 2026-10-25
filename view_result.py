from pathlib import Path
import cv2
import matplotlib.pyplot as plt

dataset = Path("dataset/DB1")
results = Path("results")

files = sorted(
    f for f in dataset.iterdir()
    if f.is_file() and f.suffix.lower() in {".tif", ".tiff"}
)

if not files:
    print("No fingerprint images found.")
    raise SystemExit

print("Total fingerprints:", len(files))

while True:
    choice = input(
        f"\nEnter fingerprint number (1-{len(files)}) or Q to quit: "
    ).strip()

    if choice.lower() == "q":
        break

    if not choice.isdigit() or not 1 <= int(choice) <= len(files):
        print("Invalid number. Try again.")
        continue

    file = files[int(choice) - 1]

    paths = [
        ("Original", file),
        ("Enhanced", results / "enhanced" / file.name),
        ("Binary", results / "binary" / file.name),
        ("Skeleton", results / "skeleton" / file.name),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Fingerprint {choice}: {file.name}")

    for ax, (title, path) in zip(axes.flat, paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        ax.set_title(title)
        ax.axis("off")

        if image is None:
            ax.text(0.5, 0.5, "Image not found",
                    ha="center", va="center")
        else:
            ax.imshow(image, cmap="gray")

    plt.tight_layout()
    plt.show()
