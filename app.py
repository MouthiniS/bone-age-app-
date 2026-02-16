import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Bone Age Assessment AI", page_icon="🦴", layout="wide")

class BoneAgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.age_head = nn.Linear(1280, 1)
        self.growth_head = nn.Linear(1280, 3)

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        growth = self.growth_head(features)
        return age.squeeze(), growth

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BoneAgeModel().to(device)
    
    # Try to load trained weights, but don't fail if they don't exist
    try:
        model.load_state_dict(torch.load("bone_age_model_final.pth", map_location=device))
        st.success("✅ Trained model loaded!")
    except:
        st.info("ℹ️ Using initialized model (demo mode)")
    
    model.eval()
    return model, device

def preprocess_image(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image_pil).unsqueeze(0)

def get_recommendation(pred_age, true_age, growth_class):
    diff = pred_age - true_age
    abs_diff = abs(diff)
    if growth_class == 0:
        status = "🔴 DELAYED MATURATION"
        rec = "⚠️ CRITICAL: Growth delay. Consult Pediatric Endocrinologist." if abs_diff > 24 else "⚠️ Monitor nutrition and Vitamin D intake"
    elif growth_class == 2:
        status = "🟠 ADVANCED MATURATION"
        rec = "⚠️ CRITICAL: Evaluate for Precocious Puberty." if abs_diff > 24 else "⚠️ Monitor growth velocity"
    else:
        status = "🟢 NORMAL MATURATION"
        rec = "✅ Normal skeletal development"
    return status, rec

st.title("🦴 Bone Age Assessment AI")
st.markdown("Advanced ML for Skeletal Maturity | MAE: 36.25 months | F1: 0.7864")

st.sidebar.markdown("## 👤 Patient Information")
patient_age = st.sidebar.number_input("Age (months)", 0, 240, 60, 1)
patient_gender = st.sidebar.radio("Gender", ["Male", "Female"])
patient_name = st.sidebar.text_input("Patient Name", "")

model, device = load_model()

col1, col2 = st.columns([1.2, 1.2])

with col1:
    st.markdown("### 📤 Upload X-Ray Image")
    uploaded_file = st.file_uploader("Upload hand/wrist X-ray", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert('RGB')
        st.image(image_pil, caption="Uploaded X-Ray", use_column_width=True)
        st.info(f"📋 File: {uploaded_file.name} | Size: {uploaded_file.size/1024:.1f} KB")

with col2:
    st.markdown("### 📊 Prediction Results")
    if uploaded_file:
        with st.spinner("🔄 Processing X-ray image..."):
            input_tensor = preprocess_image(image_pil).to(device)
            with torch.no_grad():
                pred_age, pred_growth = model(input_tensor)
                pred_age_val = pred_age.item()
                probs = F.softmax(pred_growth, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probs)
                conf = probs[pred_class] * 100
        
        st.success("✅ Analysis Complete!")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Bone Age", f"{pred_age_val:.1f} months", f"{pred_age_val-patient_age:+.1f} months")
        m2.metric("Confidence Score", f"{conf:.1f}%")
        m3.metric("Age Difference", f"{abs(pred_age_val-patient_age):.1f} months")
        
        st.markdown("**Growth Status Probabilities:**")
        for lbl, prob in zip(["Delayed (<120m)", "Normal (120-180m)", "Advanced (>180m)"], probs):
            st.progress(prob, f"{lbl}: {prob*100:.1f}%")
        
        status, rec = get_recommendation(pred_age_val, patient_age, pred_class)
        st.markdown(f"### {status}")
        if pred_class == 1:
            st.success(rec)
        else:
            st.warning(rec)

st.markdown("---")
st.markdown("## 📋 Clinical Report")

rc1, rc2 = st.columns(2)
with rc1:
    st.markdown("**Patient Information:**")
    st.text(f"Name: {patient_name or 'N/A'}\nGender: {patient_gender}\nAge: {patient_age} months\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

with rc2:
    if uploaded_file:
        st.markdown("**Assessment Findings:**")
        st.text(f"Bone Age: {pred_age_val:.1f} months\nDeviation: {pred_age_val-patient_age:+.1f} months\nStatus: {['Delayed','Normal','Advanced'][pred_class]}\nConfidence: {conf:.2f}%")

if uploaded_file:
    st.markdown("### 📊 Visualization")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].bar(['Chronological', 'Predicted Bone'], [patient_age, pred_age_val], color=['#667eea', '#764ba2'])
    axes[0].set_ylabel('Age (months)', fontweight='bold')
    axes[0].set_title('Bone Age vs Chronological Age', fontweight='bold')
    for i, v in enumerate([patient_age, pred_age_val]):
        axes[0].text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold')
    
    axes[1].pie(probs, labels=['Delayed', 'Normal', 'Advanced'], autopct='%1.1f%%', colors=['#ffeaa7', '#a8e6cf', '#ff7675'])
    axes[1].set_title('Growth Status Distribution', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

with st.expander("📖 About This Project"):
    st.markdown("""
    **Bone Age Assessment AI**
    
    - **Model**: EfficientNet-B0 (PyTorch)
    - **Task**: Predict bone age from hand X-rays
    - **Performance**: MAE 36.25 months, F1 0.7864
    - **Growth Classes**: Delayed (<120m), Normal (120-180m), Advanced (>180m)
    
    **Disclaimer**: For educational purposes only. Not for clinical diagnosis.
    """)
