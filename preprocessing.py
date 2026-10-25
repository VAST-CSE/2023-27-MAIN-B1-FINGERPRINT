import cv2
import numpy as np
from skimage.morphology import skeletonize


def preprocess_fingerprint(image):
    # 1. Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(image)

    # 2. Reduce minor noise
    denoised = cv2.GaussianBlur(
        enhanced, (3, 3), 0
    )

    # 3. Convert dark ridges into white foreground
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    # 4. Remove very small foreground components
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 10:
            cleaned[labels == i] = 255

    # 5. Skeletonize the cleaned ridge image
    skeleton = skeletonize(cleaned > 0)

    skeleton_image = (
        skeleton.astype(np.uint8) * 255
    )

    return enhanced, cleaned, skeleton_image
