# Assignment 04 #
# Canny Edge Detector #
# Ra'fat Naserdeen #

import cv2
import numpy as np

# Load Image as a grey scale image
image = cv2.imread('veg.jpg', cv2.IMREAD_GRAYSCALE)


# Creat Gaussian filter (smooth filter)
kernel_size = 5
std = 1.4



def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/ (2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

kernel = gaussian_kernel(kernel_size, std)

print(kernel)

# Noise Reduction & Compute the Gradient
# Apply a Sobel operator on x and y direction
Gx = cv2.Sobel(kernel, cv2.CV_64F, 1, 0, ksize=3)
Gy = cv2.Sobel(kernel, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the Gradient and the direction
gradient_G = np.sqrt(Gx**2 + Gy**2)
gradient_G = np.uint8(gradient_G)
angle = np.arctan2(Gy, Gx)


# Perform non-maximum suppression to thin the edges
def non_max_suppression(gradient_G, angle):
    rows = gradient_G.shape[0]
    cols = gradient_G.shape[1]
    result = np.zeros((rows, cols), dtype=np.uint8)
    angle = angle * 180 / np.pi
    angle[angle < 0] += 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            max_i = 255
            max_j = 255

            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                max_i = gradient_G[i, j+1]
                max_j = gradient_G[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                max_i = gradient_G[i+1, j-1]
                max_j = gradient_G[i-1, j+1]
            # Angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                max_i = gradient_G[i+1, j]
                max_j = gradient_G[i-1, j]
            # Angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                max_i = gradient_G[i-1, j-1]
                max_j = gradient_G[i+1, j+1]

            if (gradient_G[i, j] >= max_i) and (gradient_G[i, j] >= max_j):
                result[i, j] = gradient_G[i, j]
            else:
                result[i, j] = 0

    return result

thind_edges = non_max_suppression(gradient_G, angle)


# images display
cv2.imshow('Grey scale image', image)
cv2.imshow('Non Max Suppression ', thind_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
