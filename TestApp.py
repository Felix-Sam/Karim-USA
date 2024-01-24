import streamlit as st
from cvzone.ClassificationModule import Classifier
from PIL import Image
import numpy as np

model = Classifier(modelPath='keras_modelv2.h5', labelsPath='labelsv2.txt')

with st.sidebar:
    st.title('Violence Detection App')
    st.image('violence.png')

st.title('Drop Your image and get predictions')
image = st.file_uploader('Choose Image to Upload', type=['jpg', 'jpeg', 'png'])

if image:
    img = Image.open(image)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Convert the PIL Image to a NumPy array
    img_array = np.array(img)

    # Perform prediction
    result = model.getPrediction(img_array)
    violence = result[0][0] * 100
    non_violence = result[0][1] * 100
    if violence > non_violence:
        st.write(f"Violence Activity Detected: {violence:.4f}%")
    else:
        st.write(f"Non-Violence Activity Detected: {non_violence:.4f}%")
    st.write(result)