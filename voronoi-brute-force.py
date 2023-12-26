from math import sqrt
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt


class Pixel:
    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.color = c

    def color_to_rgb(self):
        return tuple(int(self.color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))


class Image:
    pixels = []

    def __init__(self, path, points):
        self.source = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        self.points = self.check_values(points)

    def check_values(self, points):
        for point in points:
            if point.x > self.source.shape[1]:
                print(f"Pixel {point.x} is out of bound {self.source.shape}")
                exit()
            if point.y > self.source.shape[0]:
                print(f"Pixel {point.y} is out of bound {self.source.shape}")
                exit()
        return points

    def write_and_display_image(self):
        for pixel in self.pixels:
            (r, g, b) = pixel.color_to_rgb()
            self.source[pixel.y][pixel.x] = (b, g, r)
        cv2.imshow('generated-image', self.source)
        cv2.waitKey(0)
        cv2.destroyWindow('generated-image')


def tp_brute_force(i):
    i.points.sort(key=lambda p: (p.x, p.y))
    for x in range(i.source.shape[1]):
        for y in range(i.source.shape[0]):
            max_distance = max(i.source.shape[0], i.source.shape[1])
            for point in i.points:
                site_result = []

                # Pour tester algo 1 remplacer par site-result[0]
                site_result.append(sqrt((x - point.x) ** 2 + (y - point.y) ** 2))

                # Pour tester algo 2 remplacer par site-result[1]
                site_result.append((x - point.x) + (y - point.y))

                # Pour tester algo 3 remplacer par site-result[2]
                site_result.append(max((x - point.x), (y - point.y)))
                if max_distance > site_result[0]:
                    i.pixels.append(Pixel(x, y, point.color))
                max_distance = site_result[0]


image = Image("./Lenna.png", [Pixel(50, 50, '#32a852'), Pixel(250, 250, '#3271a8'), Pixel(450, 450, '#a83232'), Pixel(300, 200, '#cb0bd9')])
tp_brute_force(image)
image.write_and_display_image()