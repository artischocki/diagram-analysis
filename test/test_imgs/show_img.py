import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to your local image file: replace with your actual file path
image_path = (
    "/home/artur/code/diagramm_text_roi_extraction/test/test_imgs/emissions.jpg"
)

# Load the image
img = mpimg.imread(image_path)

# Create a figure and axes
fig, ax = plt.subplots()
ax.imshow(img)
ax.set_title("Hover over the image to see pixel coordinates and color")
# Keep axes on to display coordinates


# Customize the coordinate display to show pixel index and RGB values
def format_coord(x, y):
    xi, yi = int(x + 0.5), int(y + 0.5)
    h, w = img.shape[:2]
    if 0 <= xi < w and 0 <= yi < h:
        color = img[yi, xi]
        return f"x={xi}, y={yi}, color={tuple(color)}"
    else:
        return f"x={x:.2f}, y={y:.2f}"


ax.format_coord = format_coord

plt.show()
