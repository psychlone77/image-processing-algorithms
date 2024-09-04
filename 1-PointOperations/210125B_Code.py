import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image, low, high):
    alpha = (high - low) / 255
    beta = low
    return np.clip(alpha * image + beta, low, high).astype(np.uint8)

# Load the image
image_path = './1-PointOperations/210125B_SrcImage.jpg'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# Convert to grayscale, by taking the mean of the RGB channels
gray_image = np.mean(image, axis=2).astype(np.uint8)

# Create a negative image
negative_image = 255 - gray_image

# Increase brightness by 20%
bright_image = np.clip(gray_image * 1.2, 0, 255).astype(np.uint8)

# Reduce image contrast (clip levels to between 125 and 175)
low_contrast_image = contrast_stretching(gray_image, 125, 175)

# Reduce image gray level depth to 4bpp (16 levels). Converted back to 8bpp for display
depth4_image = (gray_image // 16) * 16

# Vertical mirror image
vertical_mirror_image = np.fliplr(gray_image)

# Set up subplots
row1 = cv2.hconcat((gray_image, negative_image, bright_image))
row2 = cv2.hconcat((low_contrast_image, depth4_image, vertical_mirror_image))
finalframe = cv2.vconcat((row1, row2))

cv2.imwrite('./1-PointOperations/210125B_OPImage_11.jpg', gray_image)
cv2.imwrite('./1-PointOperations/210125B_OPImage_12.jpg', negative_image)
cv2.imwrite('./1-PointOperations/210125B_OPImage_13.jpg', bright_image)
cv2.imwrite('./1-PointOperations/210125B_OPImage_21.jpg', low_contrast_image)
cv2.imwrite('./1-PointOperations/210125B_OPImage_22.jpg', depth4_image)
cv2.imwrite('./1-PointOperations/210125B_OPImage_23.jpg', vertical_mirror_image)
cv2.imwrite('./1-PointOperations/210125B_SubPlot.jpg', finalframe)

# Display the subplot
plt.imshow(finalframe, cmap='gray')
plt.show()