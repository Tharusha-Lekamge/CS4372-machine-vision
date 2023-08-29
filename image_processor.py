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
