import cv2
import numpy as np

def extract_features(image):

    image = cv2.resize(image, (128, 128))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    feature_vector = gray.flatten()

    return feature_vector
