import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# ============================================================
# PCOS Ultrasound Classification App
# Deep Learning Final Project
# Model: CNN trained from scratch using TensorFlow/Keras
# Classes: noninfected / infected
# ============================================================

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="PCOS Ultrasound Classification",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Constants
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_pcos_cnn_model.keras"
IMG_SIZE = (224, 224)
DECISION_THRESHOLD = 0.10

CLASS_NAMES = {
    0: "noninfected",
    1: "infected"
}

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_cnn_model():
    """
    Load the trained CNN model.

    The model was trained using TensorFlow/Keras and saved in .keras format.
    Caching prevents reloading the model every time the user interacts with the app.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    return model


# -----------------------------
# Image Preprocessing Function
# -----------------------------
def preprocess_image(uploaded_image):
    """
    Preprocess the uploaded ultrasound image before prediction.

    Steps:
    1. Convert image to RGB.
    2. Resize image to 224 x 224.
    3. Convert image to NumPy array.
    4. Normalize pixel values to [0, 1].
    5. Add batch dimension.

    Returns:
        processed_image: image ready for CNN prediction.
    """
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array


# -----------------------------
# Prediction Function
# -----------------------------
def predict_pcos(model, image_array, threshold=DECISION_THRESHOLD):
    """
    Predict whether the ultrasound image is infected or noninfected.

    The CNN outputs a probability between 0 and 1.
    A threshold of 0.10 is used based on threshold analysis to improve recall.

    Returns:
        probability: predicted probability of infected class.
        predicted_class: 0 or 1.
        predicted_label: text label.
    """
    probability = float(model.predict(image_array, verbose=0)[0][0])
    predicted_class = 1 if probability >= threshold else 0
    predicted_label = CLASS_NAMES[predicted_class]

    return probability, predicted_class, predicted_label


# -----------------------------
# App Interface
# -----------------------------
st.title("PCOS Ultrasound Classification Using CNN")

st.markdown(
    """
    This application uses a Convolutional Neural Network (CNN) to classify ovarian ultrasound images into:

    - **infected**: PCOS case  
    - **noninfected**: normal case  

    The model was trained from scratch using **TensorFlow/Keras**.
    """
)

st.warning(
    "This application is for academic demonstration only and is not intended for real clinical diagnosis."
)

# -----------------------------
# Sidebar Information
# -----------------------------
with st.sidebar:
    st.header("Model Information")

    st.markdown(
        f"""
        **Architecture:** CNN  
        **Input size:** {IMG_SIZE[0]} × {IMG_SIZE[1]} × 3  
        **Output:** Binary classification  
        **Decision threshold:** {DECISION_THRESHOLD}  
        **Framework:** TensorFlow/Keras  
        """
    )

    st.header("Evaluation Summary")

    st.markdown(
        """
        Final evaluation at selected threshold:

        - **Accuracy:** 89.8%
        - **Precision:** 100%
        - **Recall:** 82.2%
        - **F1-score:** 90.2%
        """
    )

    st.header("Preprocessing")
    st.markdown(
        """
        Uploaded images are:
        1. Converted to RGB  
        2. Resized to 224 × 224  
        3. Normalized to [0, 1]  
        4. Passed to the CNN model  
        """
    )


# -----------------------------
# Main Upload Section
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an ultrasound image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        model = load_cnn_model()

        original_image, processed_image = preprocess_image(uploaded_file)

        st.subheader("Uploaded Ultrasound Image")
        st.image(original_image, caption="Input ultrasound image", use_container_width=True)

        probability, predicted_class, predicted_label = predict_pcos(
            model,
            processed_image,
            threshold=DECISION_THRESHOLD
        )

        st.subheader("Prediction Result")

        if predicted_class == 1:
            st.error("Prediction: infected (PCOS)")
        else:
            st.success("Prediction: noninfected")

        st.metric(
            label="Probability of infected class",
            value=f"{probability:.4f}"
        )

        st.progress(min(max(probability, 0.0), 1.0))

        st.markdown(
            f"""
            **Decision rule:**  
            If probability ≥ **{DECISION_THRESHOLD}**, the image is classified as **infected**.  
            Otherwise, it is classified as **noninfected**.
            """
        )

        st.subheader("Interpretation")

        if predicted_class == 1:
            st.markdown(
                """
                The model predicts that the image belongs to the **infected** class.  
                This means the image may show PCOS-related ultrasound features according to the trained CNN model.
                """
            )
        else:
            st.markdown(
                """
                The model predicts that the image belongs to the **noninfected** class.  
                This means the image does not strongly match the PCOS-related patterns learned by the CNN model.
                """
            )

    except Exception as e:
        st.error("An error occurred while processing the image or loading the model.")
        st.code(str(e))

else:
    st.info("Please upload an ultrasound image to start prediction.")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    **Deep Learning Final Project**  
    PCOS Ultrasound Classification Using CNN
    """
)
