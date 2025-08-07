import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------- Page Config & Background -----------------
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

st.markdown(
    """
    <style>
        .stApp {
            background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR4-1BwIAg3O5LNZoEXpQgqXsJphFBjwwzQfw&s");
            background-size: 1000px;
            background-position: center;
            background-repeat: no-repeat;
            color: white;
        }
        h1, h2, h3 {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Load Model -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('D:/DATA SCIENCE/PYTHON/Deep Learning/Tensor-flow/CNN - Image classification/cat_dog_model.h5')

model = load_model()

# ----------------- UI Heading -----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #111; font-weight: 900;'>
        🐶🐱 Cat vs Dog Classifier
    </h1>
    <p style='text-align: center; color: #333; font-size: 18px; font-weight: bold;'>
        Upload an image of a <strong>Cat</strong> or <strong>Dog</strong>, and this CNN model will predict it with confidence. <br>
        Make sure the image is clear and contains a single animal.
    </p>
    """,
    unsafe_allow_html=True
)


# ----------------- Image Upload -----------------
uploaded_file = st.file_uploader("📤 Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='🖼️ Uploaded Image', width=250)

        # Preprocess image
        img = image.resize((64, 64))
        img_array = np.array(img)
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("Predicting... ⏳"):
            prediction = model.predict(img_array)
            prediction_proba = float(prediction[0][0])

        # ----------------- Display Result -----------------
        if prediction_proba > 0.5:
            st.markdown(
                """
                <div style="background-color: white; padding: 15px; border-radius: 12px; margin-top: 15px;">
                    <h2 style="color: #ff4d4d; font-weight: bold; text-align: center;">
                        ✅ This is likely a <strong>Dog</strong>! 🐶
                    </h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background-color: white; padding: 15px; border-radius: 12px; margin-top: 15px;">
                    <h2 style="color: #66b3ff; font-weight: bold; text-align: center;">
                        ✅ This is likely a <strong>Cat</strong>! 🐱
                    </h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Show confidence score below
        st.markdown(
            f"<p style='text-align:center; color:lightgrey; font-size:18px;'>Confidence: <strong>{prediction_proba*100:.2f}%</strong></p>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error("❌ Error loading the image. Please upload a valid image file.")
        st.exception(e)
else:
    st.info("📷 Please upload a cat or dog image to begin prediction.")
