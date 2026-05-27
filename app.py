import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="GeoAI Intelligent Analyzer",
    page_icon="🌍",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------

model = load_model("geoai_cnn_model.h5")

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Import Modern Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* Global Font */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Main App Background */
.stApp {
    background-color: #F5F7FA;
    color: #111827;
}

/* Main Title */
h1 {
    color: #0F172A;
    text-align: center;
    font-size: 3.2rem;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Subheadings */
h2, h3 {
    color: #2563EB;
    font-weight: 600;
}

/* Paragraphs */
p {
    color: #374151;
    font-size: 17px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5E7EB;
}

/* Sidebar Text */
section[data-testid="stSidebar"] * {
    color: #111827;
}

/* Sidebar Box */
[data-testid="stSidebar"] .stAlert {
    background-color: #EFF6FF;
    border-radius: 12px;
    border: 1px solid #BFDBFE;
}

/* Upload Box */
[data-testid="stFileUploader"] {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 16px;
    border: 2px dashed #2563EB;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

/* Drag & Drop Area */
[data-testid="stFileUploaderDropzone"] {
    background-color: #F9FAFB;
    border-radius: 12px;
}

/* Browse Button */
[data-testid="stBaseButton-secondary"] {
    background-color: #2563EB !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
}

/* Browse Button Hover */
[data-testid="stBaseButton-secondary"]:hover {
    background-color: #1D4ED8 !important;
    color: white !important;
}

/* Success Box */
.stSuccess {
    background-color: #DCFCE7;
    color: #166534;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #86EFAC;
}

/* Info Box */
.stInfo {
    background-color: #DBEAFE;
    color: #1E3A8A;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #93C5FD;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background-color: #2563EB;
}

/* Uploaded Image */
img {
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

/* Footer */
footer {
    visibility: hidden;
}

/* Main Content Box */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Divider */
hr {
    border: 1px solid #E5E7EB;
}

</style>
""", unsafe_allow_html=True)

# ---------------- CLASS LABELS ----------------

classes = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]

# ---------------- GEOAI EXPLANATIONS ----------------

geo_explanations = {

    "AnnualCrop":
    "This region primarily contains seasonal agricultural land used for annual crop cultivation.",

    "Forest":
    "This area is dominated by dense vegetation and forest cover with minimal human settlement.",

    "HerbaceousVegetation":
    "This region contains low-growing vegetation such as grasses and herbaceous plant cover.",

    "Highway":
    "This area contains major transportation routes and road infrastructure connecting regions.",

    "Industrial":
    "This region represents industrial infrastructure including factories, warehouses, and manufacturing zones.",

    "Pasture":
    "This area is mainly used for grazing livestock and open pasture land activities.",

    "PermanentCrop":
    "This region contains long-term agricultural plantations such as orchards or vineyards.",

    "Residential":
    "This area consists of dense residential settlements and urban housing infrastructure.",

    "River":
    "This region contains flowing water bodies and river channel structures.",

    "SeaLake":
    "This area represents large water bodies such as lakes, reservoirs, or coastal sea regions."
}

# ---------------- CHATBOT FUNCTION ----------------

def geoai_chatbot(predicted_class):

    if predicted_class in ["AnnualCrop", "PermanentCrop", "Pasture"]:
        return "This land appears suitable for agriculture and farming-related activities."

    elif predicted_class in ["Forest", "HerbaceousVegetation"]:
        return "The image indicates significant vegetation and ecological coverage."

    elif predicted_class in ["Residential", "Industrial", "Highway"]:
        return "This region shows strong urban or infrastructure development."

    elif predicted_class in ["River", "SeaLake"]:
        return "This area is dominated by water-body structures and aquatic geography."

    else:
        return "GeoAI system successfully analyzed this satellite image."

# ---------------- SIDEBAR ----------------

st.sidebar.title("🌍 GeoAI System")

st.sidebar.info(
"""
AI-Powered Geo-Intelligent Conversational System

Features:
✔ Satellite Image Classification

✔ GeoAI Interpretation

✔ Explainable AI

✔ Conversational Intelligence

✔ Confidence Analysis
"""
)

# ---------------- MAIN TITLE ----------------

st.title("🌍 GeoAI Intelligent Satellite Image Analyzer")

st.markdown(
"""
### Upload a satellite image for AI-powered geospatial analysis
"""
)

# ---------------- FILE UPLOADER ----------------

uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION SECTION ----------------

if uploaded_file is not None:

    # Read Image
    image = Image.open(uploaded_file).convert("RGB")

    # Display Uploaded Image
    st.image(
        image,
        caption="🛰 Uploaded Satellite Image",
        use_container_width=True
    )

    # Convert Image
    image_array = np.array(image)

    image_resized = cv2.resize(image_array, (64, 64))

    image_resized = image_resized / 255.0

    image_input = np.expand_dims(image_resized, axis=0)

    # Prediction
    prediction = model.predict(image_input)

    predicted_index = np.argmax(prediction)

    predicted_class = classes[predicted_index]

    confidence_score = np.max(prediction) * 100

    geo_explanation = geo_explanations[predicted_class]

    chatbot_response = geoai_chatbot(predicted_class)

    # ---------------- RESULTS ----------------

    st.success("✅ GeoAI Analysis Completed Successfully!")

    st.subheader("📌 Prediction Result")

    st.write(f"### 🏷 Predicted Class: {predicted_class}")

    st.write(f"### 🎯 Confidence Score: {confidence_score:.2f}%")

    st.progress(int(confidence_score))

    st.subheader("🌍 GeoAI Interpretation")

    st.info(geo_explanation)

    st.subheader("🤖 GeoAI Assistant")

    st.success(chatbot_response)

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
"""
<center>
Developed using Deep Learning, GeoAI and Streamlit
</center>
""",
unsafe_allow_html=True
)