from tensorflow.keras.models import load_model as keras_load_model

def load_model():
    model = keras_load_model("main/bloodgroup_cnn_model.h5")  # ✅ Correct path
    return model
