import sys
from math import sqrt
from statistics import mean
import numpy as np
import random
import cv2
from copy import copy, deepcopy


class Mask:
    def __init__(self, x, y, d):
        self.x = x
        self.y = y
        self.d = d


class Color:

    def __init__(self, color, coordinates):
        self.color = color
        self.coordinates = coordinates

class Germ:

    def __init__(self):
        self.colors = []
        self.final = (0, 0, 0)

    def add_color(self, c):
        self.colors.append(c)
        return self
#Calcul de la moyenne
    def get_average(self):
        self.final = tuple(sum(x.color for x in self.colors) / len(self.colors))

#Calcul de la variance
    def get_variance(self):
        self.get_average()
        means = sum(self.final) / 3
        return sum((x - means) ** 2 for x in self.final) / len(self.final)

    def get_coords(self):
        xmin = min([x.coordinates[0] for x in self.colors])
        ymin = min([x.coordinates[1] for x in self.colors])
        xmax = max([x.coordinates[0] for x in self.colors])
        ymax = max([x.coordinates[1] for x in self.colors])
        return (xmin, xmax), (ymin, ymax)


class Pixel:

    def __init__(self):
        self.distance = sys.maxsize
        self.germ = None


class Image:

    def __init__(self, path, density):
        self.source = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.final = self.init_pixels()
        self.germs = []
        self.generate_random((0, self.source.shape[1]), (0, self.source.shape[0]), density)
        self.pixels = []

    def init_pixels(self):
        t = []
        for y in range(self.source.shape[0]):
            t.append([])
            for x in range(self.source.shape[1]):
                t[y].append(Pixel())
        return t

    def generate_random(self, w, h, p):
        (xmin, xmax) = w
        (ymin, ymax) = h
        for i in range(round(((xmax - xmin) * (ymax - ymin)) * p/100)):
            (x, y) = [
                round(random.randint(xmin, xmax - 1)),
                round(random.randint(ymin, ymax - 1))]
            self.final[y][x].distance = 0
            self.final[y][x].germ = self.add_gem(Germ().add_color(Color(self.source[y][x], (x, y))))

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
    contain_invalid_germ = True
    i = 0
    #changement de seuil possible ici mis a 10 pour rapiditer plus on ajout d'ietration plus l'image sera net
    while contain_invalid_germ and i <= 10:
        print(i)
        contain_invalid_germ = False
        img.pixels = deepcopy(img.final)
        for y, col in enumerate(img.source):
            for x, row in enumerate(img.source[y]):
                current = img.pixels[y][x]

                op_mask = None
                for mask in mask_top_to_bottom:
                    if 0 <= y + mask.y < len(img.source) and 0 <= x + mask.x < len(img.source[y + mask.y]):
                        if op_mask is None or \
                                img.pixels[y + mask.y][x + mask.x].distance < img.pixels[y + op_mask.y][x + op_mask.x].distance:
                            op_mask = mask

                if op_mask is not None and current.distance > (
                        img.pixels[y + op_mask.y][x + op_mask.x].distance + op_mask.d):
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

                if op_mask is not None and current.distance > (
                        img.pixels[y + op_mask.y][x + op_mask.x].distance + op_mask.d):
                    t = img.pixels[y + op_mask.y][x + op_mask.x]
                    current.distance = t.distance + op_mask.d
                    current.germ = t.germ

                current.germ.add_color(Color(pixel, (x, y)))

        over = 0
        bellow = 0
        for germ in img.germs:
            var = germ.get_variance()
            if var > 1000:
                contain_invalid_germ = True
                (x, y) = germ.get_coords()
                img.generate_random(x, y, 10)
                over += 1
            else:
                bellow += 1
        print(f"{over} values overs 1000 and {bellow} values bellow 1000")
        i += 1

    for y, col in enumerate(img.source):
        for x, pixel in enumerate(img.source[y]):
            img.source[y][x] = img.pixels[y][x].germ.final

image = Image("./zelda.png", 20)
voronoi(image)
image.write_and_display_image()