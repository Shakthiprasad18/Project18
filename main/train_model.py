# main/train_model.py

import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.applications import ResNet50

# Set dataset path
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')

def load_data():
    images = []
    labels = []
    for label_name in os.listdir(DATASET_DIR):
        label_path = os.path.join(DATASET_DIR, label_name)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                img_path = os.path.join(label_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (200, 200))
                    images.append(img)
                    labels.append(label_name)
    return np.array(images), np.array(labels)

# Load and preprocess data
X, y = load_data()
X = X / 255.0
X = np.expand_dims(X, axis=-1)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save label encoder
with open(os.path.join(os.path.dirname(__file__), 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(encoder, f)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(np.unique(y_encoded)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and save model
model = build_model()
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save(os.path.join(os.path.dirname(__file__), 'bloodgroup_cnn_model.h5'))
