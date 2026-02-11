import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page Configuration
st.set_page_config(page_title="Waste Classification AI", layout="centered")

st.title("♻️ Waste Classification using Deep Learning")
st.write("Upload an image to classify the waste type.")

# Constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model():
    """Load model with error handling"""
    try:
        if not os.path.exists("waste_model.h5"):
            st.error("❌ Model file 'waste_model.h5' not found!")
            st.stop()
        model = tf.keras.models.load_model("waste_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

def validate_image(uploaded_file):
    """Validate uploaded image file"""
    # Check file size
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"❌ File size exceeds {MAX_FILE_SIZE // (1024*1024)}MB limit"
    
    # Check file extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in ['jpg', 'jpeg', 'png']:
        return False, "❌ Invalid file type. Only JPG, JPEG, PNG allowed"
    
    # Try to open and verify image
    try:
        image = Image.open(uploaded_file)
        image.verify()  # Verify it's a valid image
        uploaded_file.seek(0)  # Reset file pointer after verify
        return True, "Valid image"
    except Exception as e:
        return False, f"❌ Invalid or corrupted image file: {str(e)}"

model = load_model()

class_names = [
    'battery', 'biological', 'cardboard', 'clothes',
    'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash'
]

uploaded_file = st.file_uploader("Upload Waste Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Validate image first
    is_valid, message = validate_image(uploaded_file)
    
    if not is_valid:
        st.error(message)
        st.stop()
    
    # Proceed with valid image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Classifying waste..."):
            prediction = model.predict(img_array, verbose=0)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100

        st.success(f"✅ Prediction: **{predicted_class.upper()}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
        
        # Show all predictions in expandable section
        with st.expander("📈 View All Predictions"):
            for idx, class_name in enumerate(class_names):
                prob = float(prediction[0][idx]) * 100
                st.write(f"**{class_name.capitalize()}:** {prob:.2f}%")
                st.progress(prob / 100)
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
else:
    st.info("👆 Upload an image to get started")
    with st.expander("💡 Tips for Best Results"):
        st.write("""
        - Use clear, well-lit images
        - Ensure the waste item is the main focus
        - Avoid blurry or distant shots
        - Supported formats: JPG, JPEG, PNG
        - Maximum file size: 10MB
        """)
