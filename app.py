import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="GeoAI Intelligent Satellite Image Analyzer",
    page_icon="🌍",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #f5f7fb;
    color: #111111;
}

.main {
    background-color: #f5f7fb;
}

.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-text {
    text-align: center;
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 35px;
}

/* Upload Box */
.upload-box {
    background: white;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Result Box */
.result-box {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-top: 25px;
}

/* Prediction Text */
.prediction {
    font-size: 28px;
    font-weight: 700;
    color: #2563eb;
}

/* Confidence */
.confidence {
    font-size: 20px;
    font-weight: 600;
    color: #16a34a;
}

/* Explanation */
.explanation {
    font-size: 17px;
    color: #374151;
    line-height: 1.8;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = load_model("geoai_streamlit_model.h5", compile=False)

# =========================
# CLASS LABELS
# =========================
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

# =========================
# KNOWLEDGE BASE
# =========================
geo_knowledge = {
    "AnnualCrop":
    "This region contains seasonal agricultural crop fields and farming patterns.",

    "Forest":
    "This region contains dense forest cover and rich vegetation.",

    "HerbaceousVegetation":
    "This region contains low-growing vegetation such as grasses and herbaceous plant cover.",

    "Highway":
    "This region includes road transportation networks and highway structures.",

    "Industrial":
    "This area contains industrial buildings, factories, or infrastructure.",

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

# =========================
# TITLE
# =========================
st.markdown(
    '<div class="main-title">🌍 GeoAI Intelligent Satellite Image Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-text">Upload a satellite image for AI-powered geospatial analysis</div>',
    unsafe_allow_html=True
)

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload Satellite Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="🛰 Uploaded Satellite Image",
        use_container_width=True
    )

    # Resize using PIL (NO CV2)
    image = image.resize((64, 64))

    # Convert to array
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Expand dimensions
    image_input = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_input)

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence_score = np.max(prediction) * 100

    geo_explanation = geo_knowledge[predicted_class]

    # =========================
    # RESULTS
    # =========================
    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.markdown(
        f'<div class="prediction">Predicted Class: {predicted_class}</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="confidence">Confidence Score: {confidence_score:.2f}%</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="explanation">{geo_explanation}</div>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown(
    '<div class="footer">Developed using Deep Learning, GeoAI and Streamlit</div>',
    unsafe_allow_html=True
)
