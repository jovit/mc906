import os
import cv2
import numpy as np

import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import tensorflow as tf

from skimage import io
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import os

def read_data():
    files = ["bk_0.jpg"]
    # r=root, d=directories, f = files

    images = []
    i = 0
    for f in files:
        image = cv2.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rez_image = cv2.resize(gray, (50, 50))
        images.append(rez_image)
        # images.append(rotation1)
        # images.append(rotation1)
        # images.append(rotation2)
        # images.append(noise)
        print(i)
        i += 1


    return np.array(images)

data = read_data().astype(float)
print(data)
data = data / 255.
data = np.array([it.flatten() for it in data])

model = tf.keras.models.load_model('fullmodel.h5')
model.build((1,50*50))
model.summary()

for d in data:
    output = model.predict(d.reshape((1, 50*50)))
    io.imshow((output[0] * 255.).reshape((50, 50)).astype(int), cmap='gray')
    plt.show()