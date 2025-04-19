import streamlit as st
import numpy as np
from PIL import Image
from model.eval import get_performance_metrics
from model.logistic import LogisticRegression
from model.cnn import CNN
from data_preprocessing.analyze import (
    extract_color_features,
    extract_texture_features,
    extract_line_features,
    extract_contrast_features
)

def process_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  
    st.write("Image resized to", img_array.shape)
    return img_array

def analyze_image_features(img_array):
    # Scale back to 0-255 range for feature extraction
    img_array_255 = (img_array * 255).astype(np.uint8)
    
    # Extract all features
    features = {}
    features.update(extract_color_features(img_array_255))
    features.update(extract_texture_features(img_array_255))
    features.update(extract_line_features(img_array_255))
    features.update(extract_contrast_features(img_array_255))
    return features

st.header("AI Art Detection Tool") 

tab1, tab2 = st.tabs(["1️⃣ Upload Image", "2️⃣ Detect AI Art"])

with tab1:
    st.header('Step 1: Upload an Image')

    # Upload Image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        preprocessed_image = process_image(image)
        st.session_state["preprocessed_image"] = preprocessed_image
        
        # Extract and store features
        features = analyze_image_features(preprocessed_image)
        st.session_state["image_features"] = features
        
        st.success("Image preprocessed and ready for prediction.")

    else:
        st.info("Please upload an image to begin detection.")

with tab2:
    st.header("Step 2: Choose a Model and Detect")

    model_options = ["Logistic Regression", "CNN"]

    if "preprocessed_image" not in st.session_state:
         st.warning("⚠️ Please upload and preprocess an image in Tab1 first.")
    else:
        selected_model = st.selectbox("Select a model:", model_options)
        preprocessed_image = st.session_state["preprocessed_image"]
        features = st.session_state.get("image_features", {})

        if st.button("🧠 Detect AI Art"):
            if selected_model == "CNN":
                model = CNN()
                prediction = model.predict_proba(preprocessed_image)
                ai_probability = float(prediction[0][0])
            else:
                model = LogisticRegression()
                flat_input = preprocessed_image.flatten().reshape(1, -1)
                model.initialize_parameters(flat_input.shape[1])
                ai_probability = model.predict_proba(flat_input)[0]

            # Print result
            if ai_probability is None:
                st.warning("⚠️ Prediction not available. Please make sure an image is uploaded and processed.")
            else:
                st.subheader(f"🧾 Result for: **{selected_model}**")
                st.write(f"🎯 This ART WORK is **{int(ai_probability * 100)}%** likely to be **AI-generated**.")
                st.progress(min(max(ai_probability, 0.0), 1.0))

                # Display feature analysis
                with st.expander("📊 Detailed Feature Analysis"):
                    # Color Features
                    st.subheader("🎨 Color Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("RGB Means:")
                        st.write(f"- Red: {features.get('red_mean', 0):.3f}")
                        st.write(f"- Green: {features.get('green_mean', 0):.3f}")
                        st.write(f"- Blue: {features.get('blue_mean', 0):.3f}")
                    with col2:
                        st.write("Color Ratios:")
                        st.write(f"- Red Ratio: {features.get('red_ratio', 0):.3f}")
                        st.write(f"- Green Ratio: {features.get('green_ratio', 0):.3f}")
                        st.write(f"- Blue Ratio: {features.get('blue_ratio', 0):.3f}")
                    
                    # Texture and Line Features
                    st.subheader("📐 Texture & Line Analysis")
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("Texture Metrics:")
                        st.write(f"- Entropy: {features.get('entropy', 0):.3f}")
                        st.write(f"- Edge Density: {features.get('edge_density', 0):.3f}")
                    with col4:
                        st.write("Line Features:")
                        st.write(f"- Line Count: {int(features.get('line_count', 0))}")
                        st.write(f"- Line Density: {features.get('line_density', 0):.3f}")
                    
                    # Contrast Features
                    st.subheader("🔆 Contrast Analysis")
                    st.write(f"- Michelson Contrast: {features.get('michelson_contrast', 0):.3f}")
                    st.write(f"- RMS Contrast: {features.get('rms_contrast', 0):.3f}")
                    st.write(f"- Local Contrast (3x3): {features.get('local_contrast_3x3_mean', 0):.3f}")
                    st.write(f"- Local Contrast (5x5): {features.get('local_contrast_5x5_mean', 0):.3f}")

                with st.expander("ℹ️ How this result is calculated"):
                    st.markdown("""
                    This value is based on the model's predicted probability for the AI class.
                    
                    If the model assigns a probability greater than or equal to a certain threshold (e.g., 0.5), 
                    the image is classified as AI-generated.

                    The detailed feature analysis shows various characteristics of the image that the model
                    considers when making its prediction, including:
                    - Color distribution and ratios
                    - Texture patterns and entropy
                    - Line detection and analysis
                    - Contrast measurements at different scales
                    """)

                # show the evaluation result for the model
                metrics = get_performance_metrics()
                model_metrics = metrics["logistic_regression"] if selected_model == "Logistic Regression" else metrics["cnn"]

                st.write("### Model Metrics from Evaluation Function")
                st.write("Accuracy:", f"{model_metrics['accuracy']:.2f}")
                st.write("Precision:", f"{model_metrics['precision']:.2f}")
                st.write("Recall:", f"{model_metrics['recall']:.2f}")
                st.write("F1 Score:", f"{model_metrics['f1_score']:.2f}")
                st.write("Mean Squared Error (MSE):", f"{model_metrics['mse']:.2f}")
       

