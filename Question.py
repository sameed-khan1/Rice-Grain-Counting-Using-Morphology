
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_PATH = "D:/Cmputer Vision/Assignments/CV_AI_F223106 Ass#3/RiceGrains.png"
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 400
MIN_GRAIN_AREA = 30
FOREGROUND_RATIO = 0.35

image_path = Path(__file__).resolve().parent / IMAGE_PATH
img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(
        f"Image not found: {image_path}\n"
        "Place rice.png in the same folder as this Python file."
    )

img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    21,
    5
)

kernel = np.ones((3, 3), np.uint8)

opened = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

cleaned = cv2.morphologyEx(
    opened,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)

sure_background = cv2.dilate(cleaned, kernel, iterations=2)

distance = cv2.distanceTransform(cleaned, cv2.DIST_L2, 5)

if distance.max() == 0:
    raise ValueError(
        "No grains detected. Check the image and threshold settings."
    )

_, sure_foreground = cv2.threshold(
    distance,
    FOREGROUND_RATIO * distance.max(),
    255,
    cv2.THRESH_BINARY
)

sure_foreground = sure_foreground.astype(np.uint8)
unknown = cv2.subtract(sure_background, sure_foreground)

number_of_markers, markers = cv2.connectedComponents(sure_foreground)

markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(original, markers)

result = original.copy()
grain_count = 0

for label in np.unique(markers):
    if label <= 1:
        continue

    region = np.uint8(markers == label)
    area = cv2.countNonZero(region)

    if area < MIN_GRAIN_AREA:
        continue

    x, y, w, h = cv2.boundingRect(region)
    grain_count += 1

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        1
    )

    cv2.putText(
        result,
        str(grain_count),
        (x, max(y - 4, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 0, 255),
        1
    )

result[markers == -1] = [0, 0, 255]

cv2.putText(
    result,
    f"Detected grains: {grain_count}",
    (10, 25),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 255),
    2
)

plt.figure(figsize=(12, 9))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(binary, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cleaned, cmap="gray")
plt.title("Morphologically Processed Image")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f"Final Result - Count: {grain_count}")
plt.axis("off")

plt.tight_layout()
plt.show()

print("Detected grain regions:", grain_count)
print("Note: touching grains may still be counted incorrectly.")
