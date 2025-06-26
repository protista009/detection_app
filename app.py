import streamlit as st
from PIL import Image  
import pathlib
import torch
import torchvision.transforms as transforms

pathlib.PosixPath = pathlib.WindowsPath
# Set UI layout
st.set_page_config(
    page_title="Soybean Leaf Disease Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 📌 Update these with your actual class names (replace with your Roboflow disease labels)
class_names = [
    "Mossaic Virus",           # 0
    "Southern blight",         # 1
    "Sudden Death Syndrone",   # 2
    "Yellow Mosaic",           # 3
    "bacterial_blight",        # 4
    "brown_spot",              # 5
    "crestamento",             # 6
    "ferrugen",                # 7
    "powdery_mildew",          # 8
    "septoria"                 # 9
]

# Load YOLOv5 classification model
@st.cache_resource
def load_model():
    model_path = "best.pt"
    model_data = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

    if isinstance(model_data, dict) and 'model' in model_data:
        model = model_data['model'].float().fuse().eval()
    else:
        model = model_data.eval()
    
    return model

model = load_model()

# Sidebar
st.sidebar.title("🌿 Soybean Disease Classifier")
app_mode = st.sidebar.radio("Select Page", ["Home", "About", "Classify"])

# Home
if app_mode == "Home":
    st.title("🧪 Soybean Leaf Disease Classifier")
    st.markdown("""
        Upload a leaf image, and this app will predict one of **10 possible soybean leaf disease classes.
        
        The model is based on YOLOv5 classification and trained using a labeled dataset from Roboflow.
    """)

# About
elif app_mode == "About":
    st.title("📘 About This Project")
    st.markdown("""
    - 🧠 Model: YOLOv5 Classification
    - 🏷 Classes: 10 Soybean Leaf Disease Classes
    - 🗂 Dataset: Custom dataset from Roboflow with class labels
    - 🧪 Accuracy: ~97.9% top-1
    - 🛠 Framework: PyTorch + Streamlit
    """)

# Classify
elif app_mode == "Classify":
    st.title("🔍 Classify Leaf Disease")
    uploaded_file = st.file_uploader("📤 Upload a soybean leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🧠 Predict Disease"):
            with st.spinner("Running classification..."):
                # Image preprocessing
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor()
                   
                ])

                input_tensor = transform(image).unsqueeze(0)

                with torch.no_grad():
                    prediction = model(input_tensor)
                    class_idx = prediction.argmax(1).item()
                    pred_class = class_names[class_idx]

                st.success(f"✅ Predicted Class: {pred_class}")