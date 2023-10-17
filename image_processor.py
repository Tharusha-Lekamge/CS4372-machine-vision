from PIL import Image
import numpy as np


class ImageProcessor:
    def __init__(self):
        self._initial_image = None
        self._initial_dims = None
        self._processed_image = None

        self._results = {}

    def load(self, path):
        im = Image.open(path)
        self._initial_image = im
        width, height = im.size
        self._initial_dims = (width, height)

        print("Loading image of dimensions {} x {}".format(width, height))
        return np.array(im)

    def display(self, im_array):
        """
        Input:
            im_array: Image array - accepts a numpy array
        Output:
            No returns
            Prints the image
        """
        im = Image.fromarray(im_array)
        im.show()

    def convert_to_grey(self, input_img_array, output_name: str = None):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
            output_name: Name of the output image if you want to save it
        Output:
            img_array: Image array - returns a grey-scaled image as a numpy array
            if output_name is provided, saves the image in jpg format
        """
        input_img = Image.fromarray(input_img_array)
        gray_image = input_img.convert("L")
        if output_name:
            self.save_img(gray_image, output_name)
        return np.array(gray_image)

    def rgb_to_grey(self, input_img_array, method: str = None, output_name: str = None):
        """
        Input:
            input_img_array: Image array - accepts a numpy array of RGB image
            method: Method of conversion - accepts "average", "lightness", "luminosity"
            output_name: Name of the output image if you want to save it
        Output:
            img_array: Image array - returns a grey-scaled image as a numpy array
            if output_name is provided, saves the image in jpg format
        """
        if not isinstance(input_img_array, np.ndarray):
            raise ValueError("Input image must be a NumPy array")

        # Check if the input image has 3 channels (RGB)
        if len(input_img_array.shape) != 3 or input_img_array.shape[2] != 3:
            raise ValueError("Input image must be in RGB format")

        if method == "average":
            gray_image = np.mean(input_img_array, axis=2).astype(np.uint8)
        elif method == "lightness":
            gray_image = (
                (np.max(input_img_array, axis=2) + np.min(input_img_array, axis=2)) / 2
            ).astype(np.uint8)
        elif method == "luminosity":
            gray_image = (
                0.21 * input_img_array[:, :, 0]
                + 0.72 * input_img_array[:, :, 1]
                + 0.07 * input_img_array[:, :, 2]
            ).astype(np.uint8)
        else:
            raise ValueError(
                "Method must be one of 'average', 'lightness', 'luminosity'"
            )

        if output_name:
            self.save_img(gray_image, output_name)
        return np.array(gray_image)

    def resample_img_linear_interpolation(
        self,
        array,
        output_size=None,
        resize_ratio=None,
        output_name: str = None,
    ):
        """
        input:
            array: Original image array,
            output_size: (width, height) tuple,
            resize_ratio: (width_ratio, height_ratio) tuple,
            output_name: Name of the output image if you want to save it
        output:
            resampled_image: Resampled image array
            Can save the image in jpg format if output_name is provided

        Linear interpolation algorithm
        src: https://www.youtube.com/watch?v=rLMznzIslVA
        """
        new_width = 0
        new_height = 0
        if output_size:
            resize_ratio = [
                output_size[0] / array.shape[0],
                output_size[1] / array.shape[1],
            ]
            new_width = output_size[1]
            new_height = output_size[0]
        elif resize_ratio:
            new_width = int(array.shape[1] * resize_ratio[1])
            new_height = int(array.shape[0] * resize_ratio[0])
        else:
            raise ValueError("Must provide either output size or resize ratio")

        resampled_image = np.zeros((new_height, new_width), dtype=np.uint8)

        for y in range(new_height):
            for x in range(new_width):
                original_x = x / new_width * array.shape[1]
                original_y = y / new_height * array.shape[0]
                # Get original two points
                x0, y0 = int(original_x), int(original_y)
                x1, y1 = x0 + 1, y0 + 1

                # Check if the points are out of bounds
                if x1 >= array.shape[1]:
                    x1 = x0
                if y1 >= array.shape[0]:
                    y1 = y0

                alpha = original_x - x0
                beta = original_y - y0

                resampled_image[y, x] = (
                    (1 - alpha) * (1 - beta) * array[y0, x0]
                    + alpha * (1 - beta) * array[y0, x1]
                    + (1 - alpha) * beta * array[y1, x0]
                    + alpha * beta * array[y1, x1]
                ).astype(np.uint8)
        if output_name:
            self.save_img(resampled_image, output_name)
        return resampled_image

    def canny_edge_detection(
        self,
        input_img_array,
        output_name: str = None,
        kernel_size: int = 5,
        sigma: float = 1,
        low_threshold: float = 0.05,
        high_threshold: float = 0.15,
    ):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
            output_name: Name of the output image if you want to save it
        Output:
            img_array: Image array - returns a grey-scaled image as a numpy array
            if output_name is provided, saves the image in jpg format
        """
        if not isinstance(input_img_array, np.ndarray):
            raise ValueError("Input image must be a NumPy array")

        # Check if the input image has 3 channels (RGB)
        if len(input_img_array.shape) != 3 or input_img_array.shape[2] != 3:
            raise ValueError("Input image must be in RGB format")

        # Convert to greyscale
        gray_image = self.rgb_to_grey(input_img_array, "average")
        self.display(gray_image)

        # Apply Gaussian blur
        blur_image = self.gaussian_blur(
            gray_image, kernel_size=kernel_size, sigma=sigma
        )
        self.display(blur_image)

        # Find gradients
        gradient_magnitude, gradient_direction = self.find_gradients_greyscale(
            blur_image
        )

        # Non-maximum suppression
        suppressed_image = self.non_maximum_suppression_greyscale(
            gradient_magnitude, gradient_direction
        )
        self.display(suppressed_image)

        # Double thresholding
        threshold_image = self.double_thresholding(
            suppressed_image, low_threshold, high_threshold
        )
        self.display(threshold_image)

        # Edge tracking by hysteresis
        edge_image = self.edge_tracking_by_hysteresis(threshold_image)
        self.display(edge_image)

        if output_name:
            new_p = Image.fromarray(edge_image)
            new_p = new_p.convert("L")
            self.save_img(new_p, output_name)
        return edge_image

    # Working
    def gaussian_blur(self, input_img_array, kernel_size: int = 5, sigma: float = 1):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian kernel
        Output:
            img_array: Image array - returns a grey-scaled image as a numpy array
        """
        if not isinstance(input_img_array, np.ndarray):
            raise ValueError("Input image must be a NumPy array")

        # Check if the input image has 1 channel (greyscale)
        if len(input_img_array.shape) != 2:
            raise ValueError("Input image must be in greyscale format")

        # Create a Gaussian kernel
        kernel = self.create_gaussian_kernel(kernel_size, sigma)

        # Convolve the kernel with the image
        return self.convolve_grayscale(input_img_array, kernel)

    # Working
    def create_gaussian_kernel(self, kernel_size: int = 5, sigma: float = 1):
        """
        Input:
            kernel_size: Size of the Gaussian kernel
            sigma: Standard deviation of the Gaussian kernel
        Output:
            kernel: Gaussian kernel as a numpy array
        """
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = np.exp(
                    -((i - center) ** 2 + (j - center) ** 2) / (2 * sigma**2)
                )
        print(kernel)
        return kernel / np.sum(kernel)

    # Working
    def convolve_grayscale(self, input_img_array, kernel):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
            kernel: Kernel as a numpy array
        Output:
            img_array: Image array - returns a grey-scaled image as a numpy array
        """
        if not isinstance(input_img_array, np.ndarray):
            raise ValueError("Input image must be a NumPy array")

        # Check if the input image has 3 channels (RGB)
        if len(input_img_array.shape) != 2:
            raise ValueError("Input image must be in greyscale format")

        # Check if the kernel is a square matrix
        if len(kernel.shape) != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("Kernel must be a square matrix")

        # Check if the kernel has odd dimensions
        if kernel.shape[0] % 2 == 0:
            raise ValueError("Kernel dimensions must be odd")

        # Get the center of the kernel
        center = kernel.shape[0] // 2

        # Pad the image with zeros
        padded_image = np.pad(
            input_img_array, ((center, center), (center, center)), "constant"
        )

        # Convolve the kernel with the image
        img_array = np.zeros(input_img_array.shape)
        for i in range(input_img_array.shape[0]):
            for j in range(input_img_array.shape[1]):
                img_array[i, j] = np.sum(
                    padded_image[i : i + kernel.shape[0], j : j + kernel.shape[1]]
                    * kernel
                )
        return img_array

    def find_gradients(self, input_img_array):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
        Output:
            gradient_magnitude: Image array - returns the gradient magnitude as a numpy array
            gradient_direction: Image array - returns the gradient direction as a numpy array
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Convolve the kernels with the image
        gradient_x = self.convolve(input_img_array, sobel_x)
        gradient_y = self.convolve(input_img_array, sobel_y)

        # Find the magnitude and direction of the gradient
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_magnitude, gradient_direction

    def find_gradients_greyscale(self, input_img_array):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
        Output:
            gradient_magnitude: Image array - returns the gradient magnitude as a numpy array
            gradient_direction: Image array - returns the gradient direction as a numpy array
        """
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Convolve the kernels with the image
        gradient_x = self.convolve_grayscale(input_img_array, sobel_x)
        gradient_y = self.convolve_grayscale(input_img_array, sobel_y)

        # Find the magnitude and direction of the gradient
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        gradient_direction = np.arctan2(gradient_y, gradient_x)

        return gradient_magnitude, gradient_direction

    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        # Create a padded image
        padded_image = np.pad(
            gradient_magnitude, ((1, 1), (1, 1)), "constant", constant_values=0
        )

        # Create an empty image
        suppressed_image = np.zeros(gradient_magnitude.shape)

        # Find the suppressed image
        for i in range(gradient_magnitude.shape[0]):
            for j in range(gradient_magnitude.shape[1]):
                # Get the direction of the gradient
                direction = gradient_direction[i, j]

                # Get the two pixels to compare with
                if direction <= np.pi / 8 or direction > 7 * np.pi / 8:
                    pixel1 = padded_image[i + 1, j]
                    pixel2 = padded_image[i + 1, j + 2]
                elif direction <= 3 * np.pi / 8:
                    pixel1 = padded_image[i + 1, j + 2]
                    pixel2 = padded_image[i, j + 2]
                elif direction <= 5 * np.pi / 8:
                    pixel1 = padded_image[i, j + 2]
                    pixel2 = padded_image[i, j]
                else:
                    pixel1 = padded_image[i, j]
                    pixel2 = padded_image[i + 1, j]

                # Compare the two pixels
                if (
                    gradient_magnitude[i, j] >= pixel1
                    and gradient_magnitude[i, j] >= pixel2
                ):
                    suppressed_image[i, j] = gradient_magnitude[i, j]

        return suppressed_image

    def non_maximum_suppression_greyscale(self, gradient_magnitude, gradient_direction):
        # Create a padded image
        padded_image = np.pad(
            gradient_magnitude, ((1, 1), (1, 1)), "constant", constant_values=0
        )

        # Create an empty image
        suppressed_image = np.zeros(gradient_magnitude.shape)

        # Find the suppressed image
        for i in range(gradient_magnitude.shape[0]):
            for j in range(gradient_magnitude.shape[1]):
                # Get the direction of the gradient
                direction = gradient_direction[i, j]

                # Get the two pixels to compare with
                if direction <= np.pi / 8 or direction > 7 * np.pi / 8:
                    pixel1 = padded_image[i + 1, j]
                    pixel2 = padded_image[i + 1, j + 2]
                elif direction <= 3 * np.pi / 8:
                    pixel1 = padded_image[i + 1, j + 2]
                    pixel2 = padded_image[i, j + 2]
                elif direction <= 5 * np.pi / 8:
                    pixel1 = padded_image[i, j + 2]
                    pixel2 = padded_image[i, j]
                else:
                    pixel1 = padded_image[i, j]
                    pixel2 = padded_image[i + 1, j]

                # Compare the two pixels
                if (
                    gradient_magnitude[i, j] >= pixel1
                    and gradient_magnitude[i, j] >= pixel2
                ):
                    suppressed_image[i, j] = gradient_magnitude[i, j]

        return suppressed_image

    def double_thresholding(
        self,
        suppressed_image,
        low_threshold: float = 0.05,
        high_threshold: float = 0.15,
    ):
        # Create an empty image
        threshold_image = np.zeros(suppressed_image.shape)

        # Find the thresholds
        high_threshold = suppressed_image.max() * high_threshold
        low_threshold = high_threshold * low_threshold

        # Find the threshold image
        for i in range(suppressed_image.shape[0]):
            for j in range(suppressed_image.shape[1]):
                if suppressed_image[i, j] >= high_threshold:
                    threshold_image[i, j] = 255
                elif suppressed_image[i, j] >= low_threshold:
                    threshold_image[i, j] = 128

        return threshold_image

    def edge_tracking_by_hysteresis(self, threshold_image):
        # Create a padded image
        padded_image = np.pad(
            threshold_image, ((1, 1), (1, 1)), "constant", constant_values=0
        )

        # Create an empty image
        edge_image = np.zeros(threshold_image.shape)

        # Find the edge image
        for i in range(threshold_image.shape[0]):
            for j in range(threshold_image.shape[1]):
                if padded_image[i + 1, j + 1] == 128:
                    if (
                        padded_image[i : i + 3, j : j + 3].max() == 255
                        or padded_image[i : i + 3, j : j + 3].min() == 255
                    ):
                        edge_image[i, j] = 255

        return edge_image

    def get_mean(self, input_img_array, indices=None):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
            indices: Indices of the pixels to be averaged
        Output:
            mean: Mean of the pixels
        """

        if indices is None:
            # get the average of whole image
            return np.mean(input_img_array)

        mean = 0
        for index in indices:
            mean += input_img_array[index[0], index[1]]
        return mean / len(indices)

    def get_partitions(self, input_img_array, threshold: float):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
            threshold: Threshold value for segmentation
        Output:
            high_intensity: Indices of the pixels with intensity above threshold
            low_intensity: Indices of the pixels with intensity below threshold
        """
        high_intensity = []
        low_intensity = []

        for i in range(input_img_array.shape[0]):
            for j in range(input_img_array.shape[1]):
                if input_img_array[i, j] >= threshold:
                    high_intensity.append((i, j))
                else:
                    low_intensity.append((i, j))

        return high_intensity, low_intensity

    def inter_means_segmentation(
        self, input_img_array, threshold: float = None, output_name: str = None
    ):
        """
        Input:
            input_img_array: Image array - accepts a numpy array
            threshold: Threshold value for segmentation
            output_name: Name of the output image if you want to save it
        Output:
            img_array: Image array - returns a grey-scaled image as a numpy array
            if output_name is provided, saves the image in jpg format
        """
        if not isinstance(input_img_array, np.ndarray):
            raise ValueError("Input image must be a NumPy array")

        # Check if the input image has 1 channel (greyscale)
        if len(input_img_array.shape) == 3:
            # convert to grey
            print("not in grey scale. Converting...")
            input_img_array = self.rgb_to_grey(input_img_array, "average")
            print("Converted to grey scale")

        if threshold is None:
            threshold = self.get_mean(input_img_array)

        print("initial threshold: ", threshold)

        # Create an empty image
        img_array = np.zeros(input_img_array.shape)

        high_intensity, low_intensity = self.get_partitions(input_img_array, threshold)

        print("High intensity: ", len(high_intensity))
        print("Low intensity: ", len(low_intensity))

        mu_1 = self.get_mean(input_img_array, high_intensity)
        mu_2 = self.get_mean(input_img_array, low_intensity)

        print("initial mu_1: ", mu_1)
        print("initial mu_2: ", mu_2)

        old_mu_1 = old_mu_2 = threshold

        while abs(mu_1 - old_mu_1) > 0.01 and abs(mu_2 - old_mu_2) > 0.01:
            new_threshold = (mu_1 + mu_2) / 2
            old_mu_1 = mu_1
            old_mu_2 = mu_2

            high_intensity, low_intensity = self.get_partitions(
                input_img_array, new_threshold
            )

            mu_1 = self.get_mean(input_img_array, high_intensity)
            mu_2 = self.get_mean(input_img_array, low_intensity)

        final_threshold = (mu_1 + mu_2) / 2
        print("final threshold: ", final_threshold)

        high_intensity, low_intensity = self.get_partitions(
            input_img_array, final_threshold
        )

        for index in high_intensity:
            img_array[index[0], index[1]] = 255

        # display image
        self.display(img_array)

        if output_name:
            new_p = Image.fromarray(img_array)
            new_p = new_p.convert("L")
            self.save_img(new_p, output_name)
        return img_array

    def save_img(self, img_array, output_name: str):
        if not output_name.endswith(".jpg"):
            output_name = output_name + ".jpg"
        if isinstance(img_array, Image.Image):
            img_array.save(output_name)
            return
        else:
            im = Image.fromarray(img_array)
            im.save(output_name)

    def __call__(self, path, methods_in_order):
        return self.load(path)
