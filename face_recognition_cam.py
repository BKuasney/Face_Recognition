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

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # capture frame by frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors = 5,
            minSize=(30,30),
            flags=cv2.CASCADE_SCALE_IMAGE#cv2.CV_HAAR_SCALE_IMAGE
    )

    # draw a rectangle around de faces
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+w),(0,255,0),2)

    # display the result frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# when everything is done , release the capture
video_capture.release()
cv2.destroyAllWindows()
