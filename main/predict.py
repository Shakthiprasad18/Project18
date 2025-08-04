import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from preprocessing.image_preprocess import preprocess_image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'
DATASET_DIR = "dataset"
MODEL_PATH = "main/bloodgroup_cnn_model.h5"
ENCODER_PATH = "main/label_encoder.pkl"

# Load the trained model and label encoder
try:
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print(f"Error loading model or encoder: {e}")
    model = None
    label_encoder = None

def predict_blood_group(image_path):
    """Predicts the blood group from an image path."""
    try:
        img = preprocess_image(image_path)
        img = np.expand_dims(img, axis=(0, -1))
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        return predicted_label
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model or label encoder not loaded.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    if file:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(temp_path)

        predicted_blood_group = predict_blood_group(temp_path)
        os.remove(temp_path)

        if predicted_blood_group:
            return jsonify({'blood_group': predicted_blood_group})
        else:
            return jsonify({'error': 'Error during blood group prediction.'}), 500

    return jsonify({'error': 'Invalid file upload.'}), 400

@app.route('/correct', methods=['POST'])
def correct():
    if 'image' not in request.files or 'correct_blood_group' not in request.form:
        return jsonify({'error': 'Missing image or correct blood group.'}), 400

    file = request.files['image']
    correct_blood_group = request.form['correct_blood_group'].strip()

    if file.filename == '':
        return jsonify({'error': 'No image selected.'}), 400

    correct_folder = os.path.join(DATASET_DIR, correct_blood_group)
    if not os.path.isdir(correct_folder):
        return jsonify({'error': f"The folder '{correct_blood_group}' does not exist in 'dataset'. Please check spelling."}), 400

    if file:
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(temp_path)

        count = len(os.listdir(correct_folder))
        new_filename = f"{correct_blood_group}img{count + 1}.png"
        destination = os.path.join(correct_folder, new_filename)

        try:
            shutil.copy(temp_path, destination)
            os.remove(temp_path)
            return jsonify({'message': f'Image saved to: {destination}'})
        except Exception as e:
            os.remove(temp_path)
            return jsonify({'error': f'Error saving image: {e}'}), 500

    return jsonify({'error': 'Invalid file or blood group data.'}), 400

if __name__ == '__main__':
    app.run(debug=True)