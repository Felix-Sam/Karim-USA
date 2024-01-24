from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import streamlit as st

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
# Load the model
# model = load_model("keras_modelv3.h5", compile=False)
# Load the labels
# class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
with st.sidebar:
    st.title('Violence Detection App')
    st.image('violence.png')

st.title('Drop Your image and get predictions')
image = st.file_uploader('Choose Image to Upload', type=['jpg', 'jpeg', 'png'])
if image:
    img = Image.open(image).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)
    # Convert the PIL Image to a NumPy array
    img_array = np.array(img)
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    st.write("Class:", class_name[2:], end="")
    st.write(confidence_score)
    # Print prediction and confidence score
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", confidence_score)

