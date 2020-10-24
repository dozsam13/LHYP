import random
import numpy as np


class PuzzleShuffle(object):
    def __init__(self, cut_n):
        self.cut_n = cut_n

    def __call__(self, image):
        self.cut_step = int(image.shape[0] // self.cut_n)
        self.div_offset = (image.shape[0] % self.cut_n) // 2

        shuffled_picture = np.ndarray(image.shape)
        indexes = [i for i in range(self.cut_n*self.cut_n)]
        random.shuffle(indexes)
        for i, e in enumerate(indexes):
            ix, iy = self.calculate_offset(i)
            ex, ey = self.calculate_offset(e)
            shuffledpic = shuffled_picture[ix:ix + self.cut_step, iy:iy + self.cut_step, :]
            origpic = image[ex:ex + self.cut_step, ey:ey + self.cut_step, :]

            if not shuffledpic.shape == origpic.shape:
                print(ix, iy, ex, ey)
                print(shuffledpic.shape)
                print(origpic.shape)
                print("I: ", i, " E: ", e)
                print(self.cut_step, self.div_offset)
                print("-------------------------------------")

            shuffled_picture[ix:ix + self.cut_step, iy:iy + self.cut_step, :] = origpic


        return shuffled_picture, indexes

    def calculate_offset(self, n):
        return n//self.cut_n * self.cut_step + self.div_offset, n % self.cut_n * self.cut_step + self.div_offset
