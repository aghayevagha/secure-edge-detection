
import numpy as np
import cv2

def to_grayscale(img: np.ndarray):
    R_COEF = 0.2989
    G_COEF = 0.5870
    B_COEF = 0.1140

    
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    grayscale_image = (B_COEF * b) + G_COEF * g + R_COEF * r

    return grayscale_image


def sobel_edge_detector(image, threshold = 50, blur = 1):
    for i in range(blur):
        image = apply_gaussian_blur(image)

    sobel_x_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]])
    
    sobel_y_kernel = np.array([[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]])

    rows, cols = image.shape

    sobel_x = np.zeros_like(image, dtype=float)
    sobel_y = np.zeros_like(image, dtype=float)

    for i in range(1, rows-1):
        for j in range(1, cols-1):

            roi = image[i-1:i+2, j-1:j+2]

            sobel_x[i, j] = np.sum(roi * sobel_x_kernel)
            sobel_y[i, j] = np.sum(roi * sobel_y_kernel)

    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    edges = magnitude > threshold

    return edges.astype(np.uint8) * 255, image

def gaussian_kernel(size, sigma):
    kernel_range = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(kernel_range, kernel_range)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def apply_gaussian_blur(image, ksize=5, sigma=3):

    kernel = gaussian_kernel(ksize, sigma)
    pad_width = ksize // 2
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    blurred_image = np.zeros_like(image, dtype=float)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+ksize, j:j+ksize]
            blurred_image[i, j] = np.sum(region * kernel)
    
    return blurred_image

