import cv2
import numpy as np

def load_and_resize(img_path, target_size=None):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    if target_size is not None:
        img = cv2.resize(img, target_size)
    return img

def perform_pca(image):
    h, w = image.shape
    data = np.float32(image.reshape(h * w, 1))
    mean, eigenvectors = cv2.PCACompute(data, mean=None)
    pca_image = cv2.PCAProject(data, mean, eigenvectors).reshape(image.shape).astype(np.uint8)
    return pca_image

def enhance_contrast(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Load images
mri_img_path = 'dataset/Cancer/p15/mri.jpg'
ct_img_path = 'dataset/Cancer/p15/ct.jpg' 

mri_img = load_and_resize(mri_img_path)
ct_img = load_and_resize(ct_img_path)
 
# Resize images to the same size
min_height = min(mri_img.shape[0], ct_img.shape[0])
min_width = min(mri_img.shape[1], ct_img.shape[1])
target_size = (min_width, min_height)

mri_img_resized = load_and_resize(mri_img_path, target_size)
ct_img_resized = load_and_resize(ct_img_path, target_size)

# Apply PCA and contrast enhancement
enhanced_mri = enhance_contrast(perform_pca(mri_img_resized))
enhanced_ct = enhance_contrast(perform_pca(ct_img_resized))

# Fusion (weighted average)
fused_image = cv2.addWeighted(enhanced_mri, 0.7, enhanced_ct, 0.3, 0)

# Additional contrast adjustment
fused_image = cv2.convertScaleAbs(fused_image, alpha=1, beta=0.9)

# Display images
cv2.imshow('MRI Image', mri_img)
cv2.imshow('CT Image', ct_img)
cv2.imshow('New Fused Image', fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
