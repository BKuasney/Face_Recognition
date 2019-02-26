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
