import os
os.environ["INFERENCE_SDK_SKIP_VISUALIZATION"] = "1"
import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile
import plotly.graph_objects as go


# ==============================
#  PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Sky Analyzer",
    page_icon="🌤",
    layout="wide"
)


# ==============================
#  MODERN GRADIENT UI CSS
# ==============================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* Background Gradient */
body {
    background: linear-gradient(135deg, #dbefff 0%, #f3faff 40%, #ffffff 100%) !important;
}

/* Gradient Title */
.gradient-title {
    font-size: 50px;
    font-weight: 900;
    background: linear-gradient(90deg,rgba(0, 0, 0, 1) 0%, rgba(53, 97, 156, 1) 96%, rgba(145, 181, 181, 1) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    font-size: 22px;
    color: #555;
    margin-top: -15px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.70);
    padding: 28px;
    border-radius: 18px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.08);
    backdrop-filter: blur(8px);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0072ff 0%, #00c6ff 100%);
}
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #006eff, #00c2ff);
    color: white;
    padding: 0.7rem 1.6rem;
    border-radius: 12px;
    border: none;
    font-size: 17px;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0, 114, 255, 0.35);
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.7);
    padding: 20px;
    border-radius: 15px;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
}

</style>
""", unsafe_allow_html=True)



# ==============================
#   SECURE API KEY (from secrets)
# ==============================
API_KEY = st.secrets["roboflow"]["api_key"]
WORKSPACE = st.secrets["roboflow"]["workspace"]
WORKFLOW  = st.secrets["roboflow"]["workflow"]

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)


# ==============================
#  RAIN PROBABILITY (2-letter)
# ==============================
RAIN_PROB = {
    "Cb": 0.95,   # Cumulonimbus
    "Ns": 0.90,   # Nimbostratus
    "As": 0.60,   # Altostratus
    "Sc": 0.40,   # Stratocumulus
    "Cu": 0.30,   # Cumulus
    "Ac": 0.25,   # Altocumulus
    "St": 0.20,   # Stratus
    "Cs": 0.12,   # Cirrostratus
    "Ci": 0.05,   # Cirrus
    "Unknown": 0,
    "Cc": 0.10,   # Cirrocumulus (มักไม่ทำฝน)
    "Ct": 0.15    # Custom Type (ใส่ค่า default ให้ก่อน)
}



# ==============================
#  HEADER UI
# ==============================
st.markdown('<div class="gradient-title">🌤 Sky Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based Cloud Detection & Rain Prediction Dashboard</div>', unsafe_allow_html=True)
st.write("---")



# ==============================
#  IMAGE UPLOAD
# ==============================
uploaded = st.file_uploader("Upload a sky image", type=["jpg", "png"])

def extract_predictions(result):
    """ Safe JSON extraction """
    if not isinstance(result, list) or len(result) == 0:
        return []
    if "predictions" not in result[0]:
        return []
    if "predictions" not in result[0]["predictions"]:
        return []
    return result[0]["predictions"]["predictions"]



# ==============================
#  MAIN FUNCTION
# ==============================
if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1,1])

    # Show original image
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded Image", width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    # Save file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.getvalue())
        temp_path = tmp.name

    # Run Roboflow
    result = client.run_workflow(
        workspace_name=WORKSPACE,
        workflow_id=WORKFLOW,
        images={"image": temp_path},
        use_cache=True
    )

    predictions = extract_predictions(result)
    cloud_labels = [p["class"] for p in predictions]

    if cloud_labels:
        cloud_type = max(set(cloud_labels), key=cloud_labels.count)
    else:
        cloud_type = "Unknown"

    rain_prob = RAIN_PROB.get(cloud_type, 0.10)


    # RIGHT CARD
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🌥 Detected Cloud Type")
        st.markdown(f"### **{cloud_type}**")

        # Rain Probability Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rain_prob * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {
                    "color": "red" if rain_prob > 0.7 else 
                             "orange" if rain_prob > 0.4 else 
                             "green"
                }
            },
            title={"text": "Rain Probability (%)"}
        ))

        st.plotly_chart(fig, width='content')
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")
    st.subheader("Raw Output")
    st.json(result)
