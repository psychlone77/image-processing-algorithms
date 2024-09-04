import cv2
import numpy as np

"""
210125B_Code.py
Linear Filtering - CS3713

Note: numpy only required for RMS difference calculation and correct image type conversion for 
cv2.imwrite.
Code refactored using CoPilot
"""

def initializeImage(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to grayscale
    height, width, _ = image.shape
    greyImage = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            greyImage[i][j] = int(
                image[i, j, 0] / 3 + image[i, j, 1] / 3 + image[i, j, 2] / 3
            )

    # Find the minimum and maximum pixel values
    min_val = min(min(row) for row in greyImage)
    max_val = max(max(row) for row in greyImage)

    # Apply linear contrast enhancement
    enhancedImage = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            enhancedImage[i][j] = int(
                (greyImage[i][j] - min_val) / (max_val - min_val) * 255
            )

    # Pad the image with zeros, 2 pixels on each side
    paddedImage = [[0 for _ in range(width + 4)] for _ in range(height + 4)]
    for i in range(height):
        for j in range(width):
            paddedImage[i + 2][j + 2] = enhancedImage[i][j]

    return enhancedImage, paddedImage

def applyFilter(image, filter):
    height, width = len(image), len(image[0])
    filterHeight, filterWidth = len(filter), len(filter[0])
    result = [[0 for _ in range(width - 4)] for _ in range(height - 4)]
    for i in range(2, height - 2):
        for j in range(2, width - 2):
            sum = 0
            for k in range(filterHeight):
                for l in range(filterWidth):
                    sum += image[i + k - 2][j + l - 2] * filter[k][l]
            if sum < 0:
                sum = 0
            if sum > 255:
                sum = 255
            result[i - 2][j - 2] = sum
    return result

def normalizeFilter(filter):
    filterSum = sum(sum(row) for row in filter)
    if filterSum == 0:
        return filter  # Avoid division by zero if the sum is zero
    normalizedFilter = [[value / filterSum for value in row] for row in filter]
    return normalizedFilter

def computeRMSDifference(image1, image2):
    height, width = len(image1), len(image1[0])
    diff = 0
    for i in range(height):
        for j in range(width):
            diff += (image1[i][j] - image2[i][j]) ** 2
    mean_diff = diff / (height * width)
    rms_diff = np.sqrt(mean_diff)
    return rms_diff

# Define and normalize filters
filterA = normalizeFilter([
    [0, -1, -1, -1, 0],
    [-1, 2, 2, 2, -1],
    [-1, 2, 8, 2, -1],
    [-1, 2, 2, 2, -1],
    [0, -1, -1, -1, 0],
])

filterB = normalizeFilter([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
])

filterC = normalizeFilter([
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
])

filterD = normalizeFilter([
    [0, 0, -1, 0, 0],
    [0, -2, -2, -2, 0],
    [-1, -2, 16, -2, -1],
    [0, -2, -2, -2, -1],
    [0, 0, -1, 0, 0],
])

# Initialize the image
image_path = "./2-LinearFilter/road25.png"
enhancedImage, paddedImage = initializeImage(image_path)

# Save the original enhanced image
originalImage = np.array(enhancedImage, dtype=np.uint8)
print(
    "RMS between Original Image and Itself",
    computeRMSDifference(originalImage, originalImage),
)
cv2.imwrite("./2-LinearFilter/original.jpg", originalImage)

# Apply filters and compute RMS differences
filters = [filterA, filterB, filterC, filterD]
filterNames = ["A", "B", "C", "D"]

for i, filter in enumerate(filters):
    filterImage = applyFilter(paddedImage, filter)
    print(
        f"RMS between Original Image and Filter {filterNames[i]}",
        computeRMSDifference(enhancedImage, filterImage),
    )
    filterImage = np.array(filterImage, dtype=np.uint8)
    cv2.imwrite(f"./2-LinearFilter/filter{filterNames[i]}.jpg", filterImage)