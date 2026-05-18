import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

# ============================================================
# PCOS Ultrasound Classification App
# Deep Learning Final Project - DA450
# Final Model: V3 Regularized CNN
#
# Important:
# During training:
#   infected = 0
#   noninfected = 1
#
# For medical interpretation in this app:
#   noninfected = 0
#   infected = 1
#
# Therefore:
#   raw_model_output = probability of noninfected
#   probability_infected = 1 - raw_model_output
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
# Custom Layer Used During Training
# -----------------------------
@tf.keras.utils.register_keras_serializable()
class RandomBrightness(tf.keras.layers.Layer):
    """
    Custom brightness augmentation layer used during model training.

    During inference, this layer does not modify the image.
    It is defined here so TensorFlow/Keras can load the saved model correctly.
    """

    def __init__(self, factor=0.05, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs, training=None):
        if training:
            return tf.image.random_brightness(inputs, max_delta=self.factor)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "factor": self.factor
        })
        return config


# -----------------------------
# Constants
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "v3_regularized_cnn_final.keras"

IMG_SIZE = (224, 224)

# Final selected threshold after threshold analysis
DECISION_THRESHOLD = 0.40

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
    Load the final selected V3 Regularized CNN model.

    The model contains a custom RandomBrightness layer, so it must be passed
    in custom_objects.

    compile=False is used because the app only performs prediction.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file was not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "RandomBrightness": RandomBrightness
        },
        compile=False
    )

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
    4. Add batch dimension.

    Important:
    The final V3 model already contains layers.Rescaling(1./255),
    so normalization is NOT applied again here.
    """
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.array(image).astype("float32")
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array


# -----------------------------
# Prediction Function
# -----------------------------
def predict_pcos(model, image_array, threshold=DECISION_THRESHOLD):
    """
    Predict whether the ultrasound image is infected or noninfected.

    Training class order:
        infected = 0
        noninfected = 1

    Raw sigmoid output represents class index 1:
        probability_noninfected = raw model output

    Medical interpretation:
        probability_infected = 1 - probability_noninfected

    Decision rule:
        If probability_infected >= threshold:
            infected
        else:
            noninfected
    """
    raw_output = float(model.predict(image_array, verbose=0)[0][0])

    probability_noninfected = raw_output
    probability_infected = 1.0 - probability_noninfected

    predicted_class = 1 if probability_infected >= threshold else 0
    predicted_label = CLASS_NAMES[predicted_class]

    return probability_infected, probability_noninfected, predicted_class, predicted_label


# -----------------------------
# App Interface
# -----------------------------
st.title("PCOS Ultrasound Classification Using V3 Regularized CNN")

st.markdown(
    """
    This application uses the final selected **V3 Regularized CNN** model to classify ovarian ultrasound images into:

    - **infected**: PCOS case  
    - **noninfected**: normal case  

    The model was built from scratch using **TensorFlow/Keras** and evaluated using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and threshold analysis.
    """
)

st.warning(
    "This application is for academic demonstration only and is not intended for real clinical diagnosis."
)


# -----------------------------
# Sidebar Information
# -----------------------------
with st.sidebar:
    st.header("Final Model Information")

    st.markdown(
        f"""
        **Architecture:** V3 Regularized CNN  
        **Framework:** TensorFlow/Keras  
        **Input size:** {IMG_SIZE[0]} × {IMG_SIZE[1]} × 3  
        **Model file:** `v3_regularized_cnn_final.keras`  
        **Output:** Binary classification  
        **Final threshold:** {DECISION_THRESHOLD}  
        """
    )

    st.header("Medical Label Interpretation")

    st.markdown(
        """
        During training:

        - `infected = 0`
        - `noninfected = 1`

        For medical interpretation in this app:

        - `noninfected = 0`
        - `infected = 1`

        Therefore:

        `Probability of infected = 1 - raw_model_output`
        """
    )

    st.header("Evaluation Summary")

    st.markdown(
        """
        Final evaluation after medical label correction:

        - **Accuracy:** 92.25%
        - **AUC:** 98.24%
        - **Precision:** 98.72%
        - **Recall:** 90.77%
        - **F1-score:** 94.58%
        - **Selected threshold:** 0.40
        """
    )

    st.header("Preprocessing")

    st.markdown(
        """
        Uploaded images are:

        1. Converted to RGB  
        2. Resized to 224 × 224  
        3. Passed to the V3 CNN model  

        Normalization is handled inside the model using `Rescaling(1./255)`.
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
        st.image(
            original_image,
            caption="Input ultrasound image",
            use_container_width=True
        )

        (
            probability_infected,
            probability_noninfected,
            predicted_class,
            predicted_label
        ) = predict_pcos(
            model,
            processed_image,
            threshold=DECISION_THRESHOLD
        )

        st.subheader("Prediction Result")

        if predicted_class == 1:
            st.error("Prediction: infected (PCOS)")
        else:
            st.success("Prediction: noninfected")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Probability of infected class",
                value=f"{probability_infected:.4f}"
            )

        with col2:
            st.metric(
                label="Probability of noninfected class",
                value=f"{probability_noninfected:.4f}"
            )

        st.progress(min(max(probability_infected, 0.0), 1.0))

        st.markdown(
            f"""
            **Decision rule:**  
            If the corrected probability of **infected** is greater than or equal to **{DECISION_THRESHOLD}**,  
            the image is classified as **infected (PCOS)**.  

            Otherwise, it is classified as **noninfected**.
            """
        )

        st.subheader("Interpretation")

        if predicted_class == 1:
            st.markdown(
                """
                The model predicts that the uploaded ultrasound image belongs to the **infected (PCOS)** class.

                This means the image may contain PCOS-related ultrasound patterns learned by the final V3 CNN model.
                """
            )
        else:
            st.markdown(
                """
                The model predicts that the uploaded ultrasound image belongs to the **noninfected** class.

                This means the image does not strongly match the PCOS-related patterns learned by the final V3 CNN model.
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
    **Deep Learning Final Project — DA450**  
    PCOS Ultrasound Classification Using V3 Regularized CNN  
    Built from scratch using TensorFlow/Keras
    """
)