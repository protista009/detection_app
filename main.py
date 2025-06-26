import streamlit as st
import numpy as np
import cv2
import torch
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pathlib

# Fix for Windows path issues


# Confidence thresholds
LEAF_CONF_THRESH = 0.6
HEALTH_CONF_THRESH = 0.6

# Model loading with Streamlit cache
@st.cache_resource
def load_leaf_detection_model():
    # Load YOLOv5 model with custom weights
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
    model.conf = LEAF_CONF_THRESH  # Set confidence threshold
    return model

@st.cache_resource
def load_health_model():
    return load_model('mobilenetv2_healthy_best.h5')

@st.cache_resource
def load_disease_model():
    return load_model('mobilenetv2_soybean_best_old.h5')

# Initialize models
leaf_model = load_leaf_detection_model()
health_model = load_health_model()
disease_model = load_disease_model()

# Class labels
HEALTH_LABELS = ['healthy', 'unhealthy']
DISEASE_LABELS = [
    "Healthy", "Mossaic Virus", "Southern blight", "Sudden Death Syndrome",
    "Yellow Mosaic", "bacterial_blight", "brown_spot", "ferrugen",
    "powdery_mildew", "septoria"
]

st.title('🌿 Leaf Disease Detection Pipeline')
uploaded_file = st.file_uploader("Upload leaf image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)
    st.image(image, caption='Uploaded Image', width=300, use_container_width=False)
    
    progress_bar = st.progress(0, text="Starting analysis...")
    
    # Step 1: Leaf detection - FIXED: Pass numpy array instead of tensor
    progress_bar.progress(20, text="Detecting leaves...")
    
    # Pass numpy array directly to YOLOv5 (this returns Results object with .pandas() method)
    results = leaf_model(img_array)
    
    # Process results
    detections = results.pandas().xyxy[0]
    leaf_detections = detections[detections['name'] == 'leaf']
    
    if leaf_detections.empty:
        st.error("❌ No leaves detected with sufficient confidence!")
        st.stop()
    
    # Get best leaf detection
    best_leaf = leaf_detections.iloc[0]
    x1, y1, x2, y2 = map(int, [
        best_leaf['xmin'], best_leaf['ymin'], 
        best_leaf['xmax'], best_leaf['ymax']
    ])
    leaf_confidence = best_leaf['confidence']
    leaf_crop = img_array[y1:y2, x1:x2]
    
    st.image(leaf_crop, caption=f'Detected Leaf (Confidence: {leaf_confidence:.2%})', width=200, use_container_width=False)
    
    # Step 2: Health classification
    progress_bar.progress(50, text="Checking health status...")
    
    # Preprocess for MobileNetV2
    health_img = cv2.resize(leaf_crop, (224, 224))
    if len(health_img.shape) == 2:  # Grayscale
        health_img = cv2.cvtColor(health_img, cv2.COLOR_GRAY2RGB)
    elif health_img.shape[-1] == 4:  # RGBA
        health_img = cv2.cvtColor(health_img, cv2.COLOR_RGBA2RGB)
    
    health_img = preprocess_input(health_img)
    health_pred = health_model.predict(np.expand_dims(health_img, axis=0), verbose=0)[0]
    
    health_idx = np.argmax(health_pred)
    health_status = HEALTH_LABELS[health_idx]
    health_confidence = health_pred[health_idx]
    
    if health_confidence < HEALTH_CONF_THRESH:
        st.warning(f"⚠️ Low confidence in health prediction ({health_confidence:.2%})")
    
    if health_status == 'healthy':
        st.success(f"Result: Healthy leaf 🟢 (Confidence: {health_confidence:.2%})")
        st.stop()
    
    # Step 3: Disease classification
    progress_bar.progress(80, text="Identifying disease...")
    
    # Preprocess for MobileNetV2
    disease_img = cv2.resize(leaf_crop, (224, 224))
    if len(disease_img.shape) == 2:  # Grayscale
        disease_img = cv2.cvtColor(disease_img, cv2.COLOR_GRAY2RGB)
    elif disease_img.shape[-1] == 4:  # RGBA
        disease_img = cv2.cvtColor(disease_img, cv2.COLOR_RGBA2RGB)
    
    disease_img = preprocess_input(disease_img)
    disease_pred = disease_model.predict(np.expand_dims(disease_img, axis=0), verbose=0)[0]
    
    disease_idx = np.argmax(disease_pred)
    disease = DISEASE_LABELS[disease_idx]
    disease_confidence = disease_pred[disease_idx]
    
    progress_bar.progress(100, text="Analysis complete!")
    
    # Display final results
    if disease == "Healthy":
        st.success(f"Diagnosis: Healthy leaf 🟢 (Confidence: {disease_confidence:.2%})")
    else:
        st.error(f"Diagnosis: {disease} 🔴 (Confidence: {disease_confidence:.2%})")
        st.progress(float(disease_confidence), text=f"Confidence: {disease_confidence:.2%}")
