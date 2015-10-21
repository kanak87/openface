import os
import sys
import cv2
from PIL import Image
import numpy as np
import os

fileDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(fileDir, "..", ".."))

modelDir = os.path.join(fileDir, '..', '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

sys.path.append(os.path.expanduser("~/src/dlib-18.15/python_examples"))

from openface.alignment import NaiveDlib  # Depends on dlib.

align = NaiveDlib(os.path.join(dlibModelDir, "mean.csv"),
        os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))

#img = Image.open('../data/1.png')
#img = Image.open('../data/alumnus.jpg')
img = Image.open('../data/celebrities.jpg')
buf = np.fliplr(np.asarray(img))

rgbFrame = np.zeros((img.height, img.width, 3), dtype=np.uint8)

rgbFrame[:, :, 0] = buf[:, :, 2]
rgbFrame[:, :, 1] = buf[:, :, 1]
rgbFrame[:, :, 2] = buf[:, :, 0]

cv2.imshow('frame', rgbFrame)

annotatedFrame = np.copy(rgbFrame)

bbs = align.getAllFaceBoundingBoxes(rgbFrame)
for bb in bbs:
    bl = (bb.left(), bb.bottom())
    tr = (bb.right(), bb.top())
    cv2.rectangle(annotatedFrame, bl, tr, color=(153, 255, 204),
                  thickness=2)

cv2.imshow('frame2', annotatedFrame)

while True:
    pass

print 'end'