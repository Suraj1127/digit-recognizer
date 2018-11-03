import re
import pickle
import base64
import io
from PIL import Image

import numpy as np

from .ai.utilities.util_functions import sigmoid


class DigitPredictor:

    def __init__(self, dataurl):
        self.dataurl = dataurl

        self.w = self.b = dict()


    def get_raw_image_array(self):
        imgstr = re.search(r'base64,(.*)', self.dataurl).group(1)
        image_bytes = io.BytesIO(base64.b64decode(imgstr))
        img = Image.open(image_bytes).convert('L')
        return np.array(img)
        return np.array(img).flatten().reshape(1, -1) / 255


    def get_cropped_image_array(self, image_array):
        x_1 = y_1 = x_2 = y_2 = None

        for i in range(image_array.shape[0]):
            if len(set(image_array[i, :])) != 1:
                y_1 = i
                break

        for i in range(image_array.shape[0]):
            if len(set(image_array[image_array.shape[0] - i - 1, :])) != 1:
                y_2 = image_array.shape[0] - i
                break

        for i in range(image_array.shape[1]):
            if len(set(image_array[:, i])) != 1:
                x_1 = i
                break

        for i in range(image_array.shape[1]):
            if len(set(image_array[:, image_array.shape[0] - i - 1])) != 1:
                x_2 = image_array.shape[0] - i
                break
        return image_array[y_1: y_2, x_1: x_2]

    def get_preprocessed_image_array(self):
        image_array = self.get_raw_image_array()
        cropped_image_array = self.get_cropped_image_array(image_array)
        a_t, b_t = cropped_image_array.shape
        padded_image_array = np.pad(
            cropped_image_array,

            int(0.3 * max(a_t, b_t)),
            'constant',
        )
        resized_padded_image = Image.fromarray(padded_image_array, 'L').resize((28, 28))
        # resized_padded_image.show()
        resized_cropped_image_array = np.array(resized_padded_image)
        return resized_cropped_image_array.flatten().reshape(1, -1) / 255

    def load_pickles(self, weights_path, bias_path):
        with open(weights_path, 'rb') as f:
            self.w = pickle.load(f)

        with open(bias_path, 'rb') as f:
            self.b = pickle.load(f)

    def predict(self, X):
        a = dict()
        z = dict()
        a[0] = X
        nodes_array = [self.b[i].shape[0] for i in sorted(self.b.keys())]
        for i, j in enumerate(nodes_array):
            z[i + 1] = np.matmul(a[i], self.w[i + 1].T) + self.b[i + 1].T
            a[i + 1] = sigmoid(z[i + 1])
        return np.argmax(a[i + 1], axis=1)

    def predict_digit(self):
        image_array = self.get_preprocessed_image_array()
        self.load_pickles('canvas/ai/parameters/weights.pickle', 'canvas/ai/parameters/bias.pickle')
        return self.predict(image_array)

