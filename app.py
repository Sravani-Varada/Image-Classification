import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
# Load model
model = tf.keras.models.load_model('fashion_mnist_model.h5')
# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
st.title("👗 Fashion MNIST Classifier")
st.write("Upload a 28x28 grayscale image or a colored one (auto-converted)")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","webp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Because Fashion MNIST is white on black
    image = image.resize((28, 28))
    st.image(image, caption="Input Image", use_column_width=True)

    # Prepare image
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    st.success(f"Prediction: **{class_names[class_idx]}**")
