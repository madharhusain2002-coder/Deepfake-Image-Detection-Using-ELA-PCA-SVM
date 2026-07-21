import cv2
import numpy as np
import joblib

from ela import convert_to_ela_image
from feature_extraction import extract_features

model = joblib.load("svm_model.pkl")
pca = joblib.load("pca_model.pkl")

image_path = "test.jpg"

ela = convert_to_ela_image(image_path)
features = extract_features(ela)

features = features.reshape(1, -1)
features = pca.transform(features)

prediction = model.predict(features)

if prediction[0] == 0:
    print("Prediction : REAL IMAGE")
else:
    print("Prediction : DEEPFAKE IMAGE")
