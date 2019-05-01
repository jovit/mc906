import os
import cv2
import numpy as np

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util


def random_rotation(image_array: ndarray):
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)


def random_noise(image_array: ndarray):
    return sk.util.random_noise(image_array)


def horizontal_flip(image_array: ndarray):
    return image_array[:, ::-1]


def read_data(path):
    files = []
    # r=root, d=directories, f = files
    for r, _d, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))

    images = []
    i = 0
    for f in files:
        image = cv2.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rez_image = cv2.resize(gray, (50, 50))
        flip = horizontal_flip(rez_image)
        noise = random_noise(rez_image)
        rotation1 = random_rotation(rez_image)
        rotation2 = random_rotation(rez_image)
        images.append(rez_image)
        images.append(flip)
        images.append(rotation1)
        images.append(rotation1)
        images.append(rotation2)
        images.append(noise)
        print(i)
        i += 1

    return np.array(images)
