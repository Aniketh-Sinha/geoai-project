import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="GeoAI Satellite Analyzer",
    page_icon="🌍",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
    background-color: #f5f7fa;
    color: black;
}

.main {
    background-color: #f5f7fa;
}

h1 {
    text-align: center;
    color: #0f172a;
    font-size: 42px !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #1e293b;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-size: 18px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

.block-container {
    padding-top: 2rem;
    max-width: 850px;
}

.uploadedFile {
    border-radius: 15px;
}

.result-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = load_model("geoai_clean.keras", compile=False)

# ---------------- CLASS LABELS ----------------
class_names = [
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

# ---------------- KNOWLEDGE BASE ----------------
geoai_knowledge = {
    "AnnualCrop": "This region contains seasonal agricultural crop fields.",
    "Forest": "This area is dominated by dense forest vegetation and tree cover.",
    "HerbaceousVegetation": "This region contains grasses and low-growing vegetation.",
    "Highway": "This image contains road or highway transportation infrastructure.",
    "Industrial": "This area represents industrial buildings and infrastructure.",
    "Pasture": "This region is mainly used for grazing and pasture land.",
    "PermanentCrop": "This area contains long-term plantation crops.",
    "Residential": "This area contains urban residential settlements.",
    "River": "This region contains river water bodies and channels.",
    "SeaLake": "This image represents lakes or sea water regions."
}

# ---------------- TITLE ----------------
st.markdown("<h1>🌍 GeoAI Intelligent Satellite Image Analyzer</h1>", unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align:center;'>Upload a satellite image for AI-powered geospatial analysis</h3>",
    unsafe_allow_html=True
)

st.write("")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Satellite Image",
        use_container_width=True
    )

    # Resize Image
    resized_image = image.resize((64, 64))

    # Convert to array
    image_array = np.array(resized_image)

    # Normalize
    image_array = image_array / 255.0

    # Expand dimensions
    image_input = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_input)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence_score = np.max(prediction) * 100

    geo_explanation = geoai_knowledge[predicted_class]

    # ---------------- RESULTS ----------------
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    st.markdown("## Prediction Result")

    st.success(f"Predicted Class: {predicted_class}")

    st.info(f"Confidence Score: {confidence_score:.2f}%")

    st.markdown("## GeoAI Interpretation")

    st.write(geo_explanation)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.write("")
st.write("")
st.markdown(
    "<center><b>Developed using Deep Learning, GeoAI and Streamlit</b></center>",
    unsafe_allow_html=True
)
