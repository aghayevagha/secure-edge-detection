import numpy as np
import cv2
import tenseal as ts
from sobel import *
import matplotlib.pyplot as plt
# image_size and parameters
n=30
threshold = 50

# create a TenSEAL context
context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.global_scale = 2**40
context.generate_galois_keys()

# encrypt each element individually
def encrypt_image(image, context):
    rows, cols = image.shape
    encrypted_image = np.zeros((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            encrypted_image[i, j] = ts.ckks_vector(context, [image[i, j]])
    print("encryption ended")
    return encrypted_image

# decryption
def decrypt_image(encrypted_image, context):
    rows, cols = encrypted_image.shape
    decrypted_image = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if isinstance(encrypted_image[i, j], int)==False:
                decrypted_image[i, j] = encrypted_image[i, j].decrypt()[0]
    print("decryption ended")            
    return decrypted_image

# gaussian blur
def apply_gaussian_blur_encrypted(encrypted_image, ksize=5, sigma=3):
    kernel = gaussian_kernel(ksize, sigma)
    pad_width = ksize // 2
    rows, cols = encrypted_image.shape
    padded_image = np.pad(encrypted_image, ((pad_width, pad_width), (pad_width, pad_width)), mode='constant', constant_values=0)
    blurred_image = np.zeros((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i+ksize, j:j+ksize]
            blurred_pixel = ts.ckks_vector(context, [0])
            for k in range(ksize):
                for l in range(ksize):
                    blurred_pixel += region[k, l] * kernel[k, l]
            blurred_image[i, j] = blurred_pixel
    print("blurring finished")
    return blurred_image

# generate kernel
def gaussian_kernel(size, sigma):
    kernel_range = np.linspace(-(size // 2), size // 2, size)
    x, y = np.meshgrid(kernel_range, kernel_range)
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


# Sobel operator for encrypted image
def sobel_edge_detector_encrypted(encrypted_image, context):
    #generate gradient kernels
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    rows, cols = encrypted_image.shape
    sobel_x = np.zeros((rows, cols), dtype=object)
    sobel_y = np.zeros((rows, cols), dtype=object)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            roi = np.array([[encrypted_image[i - 1, j - 1], encrypted_image[i - 1, j], encrypted_image[i - 1, j + 1]],
                            [encrypted_image[i, j - 1], encrypted_image[i, j], encrypted_image[i, j + 1]],
                            [encrypted_image[i + 1, j - 1], encrypted_image[i + 1, j], encrypted_image[i + 1, j + 1]]])

            sobel_x_vec = ts.ckks_vector(context, [0])
            sobel_y_vec = ts.ckks_vector(context, [0])

            for k in range(3):
                for l in range(3):
                    sobel_x_vec += roi[k, l] * ts.ckks_vector(context, [sobel_x_kernel[k, l]])
                    sobel_y_vec += roi[k, l] * ts.ckks_vector(context, [sobel_y_kernel[k, l]])
            
            sobel_x[i, j] = sobel_x_vec
            sobel_y[i, j] = sobel_y_vec
    sobel_x2 = np.zeros((rows, cols), dtype=object)
    sobel_y2 = np.zeros((rows, cols), dtype=object)
    # calculate square
    for i in range(rows):
        for j in range(cols):
            sobel_x2[i, j] = sobel_x[i, j] * sobel_x[i, j]
            sobel_y2[i, j] = sobel_y[i, j] * sobel_y[i, j]       
    return sobel_x2 + sobel_y2

# read image
img_loc = '/kaggle/input/smallcatimage/testimage.jpeg'
image = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (n, n))

# encrypt the grayscale image
encrypted_gray_img = encrypt_image(image, context)

# apply blurring in encrypted domain
blurred_gray_img = apply_gaussian_blur_encrypted(image)

# apply sobel operator
encrypted_sum = sobel_edge_detector_encrypted(blurred_gray_img, context)

# decrypt the results
sum = decrypt_image(encrypted_sum, context)

# Compute the magnitude and apply the threshold and convert 
magnitude = np.sqrt(sum)
edges = magnitude > threshold

# Convert edges to uint8 format for display
edges = (edges * 255).astype(np.uint8)

# Plain
plain_edges,image = sobel_edge_detector(image)


plt.figure(figsize = (15,15))

plt.subplot(1, 3, 1)
plt.imshow(np.stack([edges.astype(int)] * 3, axis=-1))

plt.subplot(1, 3, 2)
plt.imshow(np.stack([plain_edges] * 3, axis=-1))

plt.subplot(1, 3, 3)
plt.imshow(np.stack([image] * 3, axis=-1))
