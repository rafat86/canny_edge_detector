import cv2
import numpy as np

# Load Image as a grey scale image
image = cv2.imread('veg.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blurring to the image
blurred = cv2.GaussianBlur(image, (5, 5), 0)

print(blurred)
# Apply the Sobel operator to find the gradients in the x and y directions
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the magnitude and direction of the gradient
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
magnitude = np.uint8(magnitude)
direction = np.arctan2(sobel_y, sobel_x)


# Perform non-maximum suppression to thin the edges
def non_max_suppression(magnitude, direction):
    rows, cols = magnitude.shape
    result = np.zeros((rows, cols), dtype=np.uint8)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                result[i, j] = magnitude[i, j]
            else:
                result[i, j] = 0

    return result


thin_edges = non_max_suppression(magnitude, direction)

# Apply hysteresis thresholding to get the final edges
low_threshold = 0
high_threshold = 200
edges = cv2.Canny(thin_edges, low_threshold, high_threshold)


print(blurred)
# Display the edges
cv2.imshow('Edges', thin_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
