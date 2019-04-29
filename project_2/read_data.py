import os
import cv2
import numpy as np

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
        rez_image = cv2.resize(gray,(50,50))
        images.append(rez_image)
        print(i)
        i += 1

    return np.array(images)
