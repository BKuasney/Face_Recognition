import cv2
import logging
import sys
import os

'''
    The XML document it's a default document for opencv to identify frontal faces
    There are many of XML documents to identify a lot of things, like cars, body humans, etc
'''

# configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.getLogger(__name__)


# load face cascade on memory
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# read image and convert to gray scale
# many operations in opencv are done in grayscale
image = cv2.imread('demo_3.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1, #this parameter means the accuracy basiclly, on error then increase for test
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

logging.info('Found {} faces!'.format(len(faces)))

# Draw a rectanglearound the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)


cv2.imshow("Faces found", image)
cv2.resize(image,(600,600))
cv2.waitKey(0)
