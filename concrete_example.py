import time
import numpy as np
from concrete import fhe

def fhe_sobel_edge_detector(image):
    #image = apply_gaussian_blur(image)
    sobel_x_kernel = np.array([[[[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]]]])

    sobel_y_kernel = np.array([[[[-1, -2, -1],
                               [0, 0, 0],
                               [1, 2, 1]]]])


    with fhe.tag("convolutions"):
        sobel_x = fhe.conv(image, sobel_x_kernel)
        sobel_y = fhe.conv(image, sobel_y_kernel)

    with fhe.tag("magnitude"):
        scaled_square = fhe.univariate(lambda x: (x**2) // 10)

        with fhe.tag("scale/square"):
            scaled_squared_x = scaled_square(sobel_x)
            scaled_squared_y = scaled_square(sobel_y)

        with fhe.tag("sqrt"):
            magnitude = (np.sqrt(scaled_squared_x + scaled_squared_y) * np.sqrt(10)).astype(np.int64)

    with fhe.tag("comparison"):
        threshold = 50
        edges = magnitude > threshold

    return edges * 255

inputset = fhe.inputset(fhe.tensor[fhe.uint6, 1, 1, 148, 148])
configuration = fhe.Configuration(
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=".keys",
    show_progress=True,
    progress_tag=True,
)

compiler = fhe.Compiler(fhe_sobel_edge_detector, {"image": "encrypted"})
circuit = compiler.compile(inputset, configuration, verbose=True)

print("Key generating...")
circuit.keygen()
print("Key generated.")

print("Running...")
start = time.time()
circuit.encrypt_run_decrypt(*inputset[0])
end = time.time()
print(f"Took {end - start} seconds")