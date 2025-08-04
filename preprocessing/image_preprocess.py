import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ Could not read image from path: {image_path}")
    img = cv2.resize(img, (200, 200))
    img = img / 255.0
    return img
