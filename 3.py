import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def load_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    return img

def apply_pca(image):
    data = np.float32(image.reshape(-1, 1))
    mean, eigenvectors = cv2.PCACompute(data, mean=None)
    pca_image = cv2.PCAProject(data, mean, eigenvectors).reshape(image.shape).astype(np.uint8)
    return cv2.normalize(pca_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def fuse_images(mri, ct):
    # Resize CT image to match MRI image size
    ct_resized = cv2.resize(ct, (mri.shape[1], mri.shape[0]))

    # Flatten the images
    mri_flat = mri.flatten()
    ct_flat = ct_resized.flatten()

    # Combine the flattened images into a single matrix
    combined_matrix = np.vstack((mri_flat, ct_flat)).T

    # Apply PCA
    pca = PCA(n_components=1)
    fused_image_flat = pca.fit_transform(combined_matrix)

    # Reshape the fused image to the original size
    fused_image = fused_image_flat.reshape(mri.shape)
    return fused_image.astype(np.uint8)

def extract_hist_features(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

# Dataset path and categories
dataset_path = 'dataset'
categories = ['Cancer', 'No cancer']

X, y = [], []
fused_cancer_images = []

for category in categories:
    category_path = os.path.join(dataset_path, category)
    for patient_folder in os.listdir(category_path):
        patient_path = os.path.join(category_path, patient_folder)
        try:
            mri = load_image(os.path.join(patient_path, 'mri.jpg'))
            ct = load_image(os.path.join(patient_path, 'ct.jpg'))
        except FileNotFoundError as e:
            print(e)
            continue

        fused_image = fuse_images(mri, ct)

        if category == 'No cancer' and len(fused_cancer_images) < 5:
            fused_cancer_images.append(fused_image)

        features = extract_hist_features(fused_image)
        X.append(features)
        y.append(1 if category == 'Cancer' else 0)

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred_train = svm.predict(X_train)
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy}")

y_pred_test = svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy}")

for i, fused_img in enumerate(fused_cancer_images):
    plt.subplot(1, 5, i + 1)
    plt.imshow(fused_img, cmap='gray')
    plt.title(f'Cancer {i+1}')
    plt.axis('off')

plt.show()
