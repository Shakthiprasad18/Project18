import os
import numpy as np
from preprocessing.image_preprocess import preprocess_image
from sklearn.preprocessing import LabelEncoder

def load_data(dataset_path):
    images = []
    labels = []
    classes = os.listdir(dataset_path)
    
    for label in classes:
        folder = os.path.join(dataset_path, label)
        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(label)
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    X = np.array(images).reshape(-1, 200, 200, 1)
    return X, y, encoder
