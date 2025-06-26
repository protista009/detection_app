import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import pathlib
import sys

# Fix for Windows path issues
pathlib.PosixPath = pathlib.WindowsPath
sys.path.append("yolov5")

# Set UI layout
st.set_page_config(
    page_title="Soybean Leaf Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Class labels
class_names = ["leaf", "not_leaf"]

# Confidence threshold for "leaf"
MIN_CONFIDENCE = 0.75  # 75%

@st.cache_resource
def load_model():
    model_path = "best.pt"
    model_data = torch.load(model_path, map_location=torch.device("cpu"))
    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model'].float().eval()
    else:
        model = model_data.eval()
    return model

model = load_model()

# Sidebar navigation
st.sidebar.title("🌿 Soybean Leaf Classifier")
app_mode = st.sidebar.radio("Select Page", ["Home", "About", "Classify"])

# Home Page
if app_mode == "Home":
    st.title("🧪 Soybean Leaf Classifier")
    st.markdown("""
        Upload a soybean leaf image, and this app will classify it as either a **leaf** or **not_leaf**.
        
        The model is based on YOLOv5 classification and trained on a Roboflow dataset.
    """)

# About Page
elif app_mode == "About":
    st.title("📘 About This Project")
    st.markdown("""
    - 🧠 **Model**: YOLOv5 Classification
    - 🏷️ **Classes**: 2 (leaf, not_leaf)
    - 🗂️ **Dataset**: Custom Roboflow Dataset
    - 🎯 **Accuracy**: ~97.9%
    - ⚙️ **Framework**: PyTorch + Streamlit
    """)

# Classify Page
elif app_mode == "Classify":
    st.title("🔍 Classify Image")
    uploaded_file = st.file_uploader("📤 Upload a soybean leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🧠 Predict"):
            with st.spinner("Classifying..."):
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                ])
                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_tensor)

                # Handle tuple output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # Classification model output
                if outputs.dim() == 2 and outputs.shape[1] == len(class_names):
                    probabilities = F.softmax(outputs, dim=1)
                    class_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[0, class_idx].item()
                    pred_class = class_names[class_idx]

                    if pred_class == "leaf":
                        if confidence >= MIN_CONFIDENCE:
                            st.success(f"✅ Prediction: {pred_class} (Confidence: {confidence:.2%})")
                        else:
                            st.warning(f"⚠️ Model predicts 'leaf' but with low confidence ({confidence:.2%}). Not enough certainty to confirm.")
                    else:
                        st.success(f"✅ Prediction: {pred_class} (Confidence: {confidence:.2%})")

                # Detection-style model output (fallback)
                elif outputs.dim() == 3 or (outputs.dim() == 2 and outputs.shape[1] >= 6):
                    detections = outputs[0] if outputs.dim() == 3 else outputs
                    if detections.numel() > 0:
                        confidences = detections[:, 4]
                        max_conf_idx = torch.argmax(confidences)
                        best_detection = detections[max_conf_idx]
                        if best_detection.numel() >= 6:
                            class_idx = int(best_detection[5].item())
                            confidence = best_detection[4].item()
                            pred_class = class_names[class_idx]
                            if pred_class == "leaf":
                                if confidence >= MIN_CONFIDENCE:
                                    st.success(f"✅ Prediction: {pred_class} (Confidence: {confidence:.2%})")
                                else:
                                    st.warning(f"⚠️ Model predicts 'leaf' but with low confidence ({confidence:.2%}). Not enough certainty to confirm.")
                            else:
                                st.success(f"✅ Prediction: {pred_class} (Confidence: {confidence:.2%})")
                        else:
                            st.error("⚠️ Invalid detection format: Missing class index")
                    else:
                        st.warning("🔍 No objects detected in the image")
                else:
                    st.error(f"❌ Unsupported model output format. Dimensions: {outputs.dim()}, Shape: {outputs.shape}")
