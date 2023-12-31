import math
import sys
from math import sqrt
from statistics import mean

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


class Mask:
    def __init__(self, x, y, d):
        self.x = x
        self.y = y
        self.d = d


class Germ:

    def __init__(self):
        self.colors = []
        self.final = (0, 0, 0)

    def add_color(self, c):
        self.colors.append(c)
        return self

    def get_average(self):
        c = list(zip(*self.colors))
        self.final = (
            round(mean(c[0])),
            round(mean(c[1])),
            round(mean(c[2]))
        )


class Pixel:

    def __init__(self):
        self.distance = sys.maxsize
        self.germ = None


class Image:

    def __init__(self, path, percentage):
        self.source = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.pixels = self.init_pixels()
        self.p = percentage / 100
        self.germs = []
        self.generate_random()

    def init_pixels(self):
        t = []
        for y in range(self.source.shape[0]):
            t.append([])
            for x in range(self.source.shape[1]):
                t[y].append(Pixel())
        return t

    def generate_random(self):
        for i in range(round((self.source.shape[0] * self.source.shape[1]) * self.p)):
            (x, y) = [
                round(random.randint(0, self.source.shape[1] - 1)),
                round(random.randint(0, self.source.shape[0] - 1))]
            self.pixels[y][x].distance = 0
            self.pixels[y][x].germ = self.add_gem(Germ().add_color(self.source[y][x]))

    def add_gem(self, germ):
        self.germs.append(germ)
        return germ

    def write_and_display_image(self):
        cv2.imshow('generated-image', self.source)
        cv2.waitKey(0)
        cv2.destroyWindow('generated-image')


mask_top_to_bottom = [
    Mask(-1, 0, 1),
    Mask(-1, -1, sqrt(2)),
    Mask(0, -1, 1),
    Mask(1, -1, sqrt(2))
]

mask_bottom_to_top = [
    Mask(1, 0, 1),
    Mask(1, 1, sqrt(2)),
    Mask(0, 1, 1),
    Mask(-1, 1, sqrt(2))
]


def voronoi(img):
    for y, col in enumerate(img.source):
        for x, row in enumerate(img.source[y]):
            current = img.pixels[y][x]

            op_mask = None
            for mask in mask_top_to_bottom:
                if 0 <= y + mask.y < len(img.source) and 0 <= x + mask.x < len(img.source[y + mask.y]):
                    if op_mask is None or \
                            img.pixels[y + mask.y][x + mask.x].distance < img.pixels[y + op_mask.y][x + op_mask.x].distance:
                        op_mask = mask

            if op_mask is not None and current.distance > (img.pixels[y + op_mask.y][x + op_mask.x].distance + op_mask.d):
                t = img.pixels[y + op_mask.y][x + op_mask.x]
                current.distance = t.distance + op_mask.d
                current.germ = t.germ

    for y, col in reversed(list(enumerate(img.source))):
        for x, pixel in reversed(list(enumerate(img.source[y]))):
            current = img.pixels[y][x]

            op_mask = None
            for mask in mask_bottom_to_top:
                if 0 <= y + mask.y < len(img.source) and 0 <= x + mask.x < len(img.source[y + mask.y]):
                    if op_mask is None or \
                            img.pixels[y + mask.y][x + mask.x].distance < img.pixels[y + op_mask.y][x + op_mask.x].distance:
                        op_mask = mask

            if op_mask is not None and current.distance > (img.pixels[y + op_mask.y][x + op_mask.x].distance + op_mask.d):
                t = img.pixels[y + op_mask.y][x + op_mask.x]
                current.distance = t.distance + op_mask.d
                current.germ = t.germ

            current.germ.add_color(pixel)

    for germ in img.germs:
        germ.get_average()

    for y, col in enumerate(img.source):
        for x, pixel in enumerate(img.source[y]):
            img.source[y][x] = img.pixels[y][x].germ.final


image = Image("./zelda.png", 5)
voronoi(image)
image.write_and_display_image()