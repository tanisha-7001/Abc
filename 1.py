import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def pca(image):
    data = np.float32(image.flatten().reshape(-1, 1))
    mean, eigenvectors = cv2.PCACompute(data, mean=None)
    pca = cv2.PCAProject(data, mean, eigenvectors)
   
    return pca.reshape(image.shape).astype(np.uint8)


def enhance_contrast(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def fuse(mri, ct):
    mri = cv2.imread(mri , cv2.IMREAD_GRAYSCALE)
    ct = cv2.imread(ct , cv2.IMREAD_GRAYSCALE)

    height = min(mri.shape[0], ct.shape[0])
    width = min(mri.shape[1], ct.shape[1])
    size = (width, height)

    mri = cv2.resize(mri, size)
    ct = cv2.resize(ct, size)

    mri = pca(mri)
    ct = pca(ct)

    mri = enhance_contrast(mri)
    ct = enhance_contrast(ct)

    fused_image = cv2.addWeighted(mri, 0.7, ct, 0.3, 0)
    fused_image = cv2.convertScaleAbs(fused_image, alpha=1, beta=0.9)
    return fused_image

def hist_features(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

dataset = 'dataset'
category = ['Cancer', 'No cancer']
X, y = [], []

for c in category:
    path1 = os.path.join(dataset , c )
    for label in os.listdir(path1):
        path2 = os.path.join(path1, label)
        mri = os.path.join(path2, 'mri.jpg')
        ct = os.path.join(path2, 'ct.jpg')

        fused_image = fuse(mri, ct)
    
        features = hist_features(fused_image)
        label = 1 if c  == 'Cancer' else 0
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2,)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_train_prediction = svm.predict(X_train)
train_score = accuracy_score(y_train, y_train_prediction)
print('Training Accuracy:',train_score)

y_test_prediction = svm.predict(X_test)
test_score = accuracy_score(y_test, y_test_prediction)
print('Test Accuracy: ',test_score)

