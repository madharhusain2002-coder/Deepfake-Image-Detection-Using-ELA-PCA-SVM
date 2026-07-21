import os
import cv2
import numpy as np

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ela import convert_to_ela_image
from feature_extraction import extract_features

real_path = "dataset/real"
fake_path = "dataset/fake"

X = []
y = []

for folder, label in [(real_path,0),(fake_path,1)]:

    if not os.path.exists(folder):
        continue

    for file in os.listdir(folder):

        path = os.path.join(folder,file)

        try:
            ela = convert_to_ela_image(path)
            feature = extract_features(ela)

            X.append(feature)
            y.append(label)

        except:
            pass

X = np.array(X)
y = np.array(y)

pca = PCA(n_components=100)

X = pca.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

model=SVC(kernel="linear")

model.fit(X_train,y_train)

pred=model.predict(X_test)

print("Accuracy :",accuracy_score(y_test,pred))
