import os
import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision import models, transforms

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Cassava AI Detection",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #f4efe6;
        color: #2b2119;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1f1813;
        margin-bottom: 0.3rem;
    }
    .subtext {
        font-size: 1.1rem;
        color: #6d5848;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        background: #e2d2b3;
        color: #5a422f;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .result-card {
        background: #fffaf2;
        border: 1px solid #e0d2ba;
        border-radius: 1rem;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }
    .result-label {
        font-size: 1.3rem;
        font-weight: 800;
        color: #1f1813;
    }
    .result-meta {
        color: #8b735d;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .result-desc {
        color: #6d5848;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-top: 0.5rem;
    }
    .confidence {
        color: #b45309;
        font-weight: 700;
        margin-top: 0.65rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cpu"
IMG_SIZE = 224
MODEL_PATH = "outputs/best_efficientnet_b0.pth"
CONFIDENCE_THRESHOLD = 90.0

CLASS_NAMES = [
    "Cassava Bacterial Blight",
    "Cassava Brown Streak Disease",
    "Cassava Green Mottle",
    "Cassava Mosaic Disease",
    "Healthy"
]

DESCRIPTIONS = {
    "Cassava Bacterial Blight": "A bacterial disease that may cause wilting, angular leaf spots, and blight symptoms that weaken plant health and reduce productivity.",
    "Cassava Brown Streak Disease": "A viral disease associated with leaf chlorosis and root damage, which can lower market quality and overall yield.",
    "Cassava Green Mottle": "A leaf condition marked by mottled green patterns that may indicate abnormal development and possible stress on the crop.",
    "Cassava Mosaic Disease": "A viral disease recognized by mosaic-like patches and leaf distortion, often leading to reduced growth and lower yield.",
    "Healthy": "The leaf appears healthy, with no major visual signs of the cassava disease classes included in the model.",
    "Not a valid cassava leaf": "The uploaded image is either not a cassava leaf or the model confidence is too low to provide a reliable prediction."
}

# -----------------------------
# MODEL
# -----------------------------
def build_model(num_classes=5):
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

@st.cache_resource
def load_model():
    model = build_model(len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = CLASS_NAMES[pred.item()]
    score = confidence.item() * 100
    return label, score, probs.squeeze(0).cpu().numpy()

# -----------------------------
# UI
# -----------------------------
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="badge">Cassava AI Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Detect Cassava Diseases Instantly</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtext">Upload a cassava leaf image and let the system identify the condition and provide a short explanation based on the detected class.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a cassava leaf image",
        type=["jpg", "jpeg", "png"]
    )

with right:
    st.markdown("### 🌿 Cassava Leaf Preview")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)

        with st.spinner("Running model..."):
            raw_label, confidence, probs = predict_image(image)

        if confidence < CONFIDENCE_THRESHOLD:
            label = "Not a valid cassava leaf"
            st.error("Prediction confidence is below 95%. This image may not be a valid cassava leaf.")
        else:
            label = raw_label

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-meta">Detected Condition</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-label">{label}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-desc">{DESCRIPTIONS[label]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Class Probabilities")
        prob_data = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}
        st.bar_chart(prob_data)
    else:
        st.info("Upload an image to start detection.")
