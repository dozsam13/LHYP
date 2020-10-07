import random
import numpy as np


class PuzzleShuffle(object):
    def __call__(self, image):
        shuffled_picture = np.ndarray(image.shape)
        indexes = [i for i in range(4)]
        random.shuffle(indexes)
        for i, e in enumerate(indexes):
            ix, iy = self.calculate_offset(i)
            ex, ey = self.calculate_offset(e)
            shuffled_picture[ix:ix + 55, iy:iy + 55, :] = image[ex:ex + 55, ey:ey + 55, :]

        return shuffled_picture, indexes

    def calculate_offset(self, n):
        return n//2 * 55, n%2 * 55
