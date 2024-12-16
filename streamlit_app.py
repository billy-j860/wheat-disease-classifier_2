import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Wheat Classifier",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Function to set custom background style
def set_custom_background():
    style = """
        <style>
        .stApp {
            background-color: #f5f5f5;
            background-image: linear-gradient(to right, rgba(0,0,0,0.02) 5%, transparent 5%),
                            linear-gradient(to bottom, rgba(0,0,0,0.02) 5%, transparent 5%);
            background-size: 20px 20px;
            background-position: center;
            background-repeat: repeat;
            background-attachment: fixed;
        }
        .stButton>button {
            background-color: #69b34c;
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            border: none;
            box-shadow: 0 3px 5px rgba(0, 0, 0, 0.1);
            font-weight: bold;
            transition: 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #498c34;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
        }
        .upload-section, .results-section {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .title-text {
            text-align: center;
            color: #1D4F6F;
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .subtitle-text {
            text-align: center;
            color: #2E6B8E;
            font-size: 1.5rem;
            margin-bottom: 2rem;
        }
        footer {
            text-align: center;
            padding: 20px;
            background-color: white;
            border-radius: 10px;
            margin-top: 2rem;
        }
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Set the custom background
set_custom_background()

def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('models/best_efficientnet_final.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for model prediction"""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error during image preprocessing: {str(e)}")
        return None

@st.cache_data
def get_prediction(model, _image):
    """Get model prediction for the image"""
    try:
        processed_image = preprocess_image(_image)
        if processed_image is not None:
            with st.spinner("Processing image..."):
                prediction = model.predict(processed_image, verbose=0)
            return prediction
        return None
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

# class names
CLASS_NAMES = ['Aphid', 'Yellow Rust', 'Fusarium Head Blight', 'Mildew', 'Healthy']
CLASS_DESCRIPTIONS = {
    'Aphid': 'Small sap-sucking insects that can cause significant damage to wheat crops.',
    'Yellow Rust': 'A fungal disease causing yellow stripes on leaves, reducing crop yield.',
    'Fusarium Head Blight': 'A serious fungal disease affecting wheat heads and grains.',
    'Mildew': 'A fungal disease appearing as a white powdery coating on leaves.',
    'Healthy': 'Normal, disease-free wheat plant.'
}

def main():
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.markdown("Explore:")
        st.markdown("- 📁 Upload your wheat images")
        st.markdown("- 📊 View prediction results")
        st.markdown("- 🌾 Learn about diseases")
        
        # disease information in sidebar
        st.markdown("---")
        st.markdown("### Disease Information")
        for class_name, description in CLASS_DESCRIPTIONS.items():
            with st.expander(class_name):
                st.write(description)

    # Main content
    st.markdown('<p class="title-text">🌾 Wheat Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Detect Pests and Diseases with AI</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check if the model file exists.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["Upload & Analyze", "Results"])
    
    with tab1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose a wheat image...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.subheader("Analysis Results")
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Analyze Image"):
                    with st.spinner("Analyzing... Please wait."):
                        prediction = get_prediction(model, image)
                        
                        if prediction is not None:
                            # Create DataFrame for visualization
                            data = pd.DataFrame({
                                "Class": CLASS_NAMES,
                                "Confidence (%)": [p * 100 for p in prediction[0]]
                            })
                            
                            # Create bar chart
                            fig = px.bar(data, 
                                       x="Confidence (%)", 
                                       y="Class", 
                                       orientation="h",
                                       text="Confidence (%)",
                                       color="Confidence (%)",
                                       color_continuous_scale="RdYlGn")
                            
                            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            fig.update_layout(yaxis={'categoryorder':'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display prediction result
                            predicted_class = CLASS_NAMES[np.argmax(prediction)]
                            confidence = np.max(prediction) * 100
                            
                            if confidence > 70:
                                st.success(f"Prediction: **{predicted_class}**")
                                st.markdown(f"**Description**: {CLASS_DESCRIPTIONS[predicted_class]}")
                            else:
                                st.warning(f"Prediction: **{predicted_class}** (Low confidence)")
                            
                            st.info(f"Confidence: {confidence:.1f}%")
                            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.error("Please make sure the uploaded file is a valid image file.")
        else:
            st.write("Please upload an image to analyze.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # About section
    with st.expander("About the Model"):
        st.write("""
        This application uses a deep learning model trained to identify common wheat diseases and pests.
        
        The model can detect:
        - Aphids (Pest)
        - Yellow Rust (Disease)
        - Fusarium Head Blight (Disease)
        - Mildew (Disease)
        - Healthy Plants
        
        For best results:
        - Upload clear, well-lit images
        - Ensure the affected area is clearly visible
        - Try to minimize background noise in the image
        
        We are continuously improving our model to provide the best service for you.
        """)
    
    # Footer
    st.markdown("""
        <footer>
            <p>Wheat Disease Classification Project</p>
            <p>© 2024 All rights reserved</p>
        </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
