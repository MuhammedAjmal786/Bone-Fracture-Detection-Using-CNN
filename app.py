# Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# title
st.title('Bone Fracture Detection')

# Upload image
imag = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])

if imag is not None:
    # Load and preprocess the image
    img = image.load_img(imag, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Load the model
    model = tf.keras.models.load_model("Bone_Fracture_Classifier.h5")
    # Make predictions
    prediction = model.predict(img_array)

    # Submit button and display result
    if st.button('Submit'):
        if prediction > 0.5:  
            st.success("Not Fractured")
        else:
            st.error("Fractured")