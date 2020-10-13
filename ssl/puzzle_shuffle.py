import random
import numpy as np


class PuzzleShuffle(object):
    def __init__(self, cut_n, img_size):
        self.cut_n = cut_n
        self.cut_step = int(img_size / cut_n)

    def __call__(self, image):
        shuffled_picture = np.ndarray(image.shape)
        indexes = [i for i in range(self.cut_n*self.cut_n)]
        random.shuffle(indexes)
        random.shuffle(indexes)
        for i, e in enumerate(indexes):
            ix, iy = self.calculate_offset(i)
            ex, ey = self.calculate_offset(e)
            shuffled_picture[ix:ix + self.cut_step, iy:iy + self.cut_step, :] = image[ex:ex + self.cut_step, ey:ey + self.cut_step, :]

        return shuffled_picture, indexes

    def calculate_offset(self, n):
        return n//2 * self.cut_step, n%2 * self.cut_step
