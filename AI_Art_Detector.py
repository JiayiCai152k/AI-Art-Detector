import streamlit as st
import numpy as np
from PIL import Image
from model.eval import get_performance_metrics
from model.logistic import LogisticRegression, predict_single_image
from model.cnn import CNN

def process_image(img):
    img = img.convert('RGB')
    # Create uint8 version for OpenCV operations (0-255)
    img_array_uint8 = np.array(img, dtype=np.uint8)
    # Create float32 version for neural network (0-1)
    img_array_float = (img_array_uint8 / 255.0).astype(np.float32)
    
    st.write("Image dimensions:", img_array_uint8.shape)
    return {
        'uint8': img_array_uint8,
        'float32': img_array_float
    }
    

st.header("AI Art Detection Tool") 

tab1, tab2 = st.tabs(["1️⃣ Upload Image", "2️⃣ Detect AI Art"])

with tab1:
    st.header('Step 1: Upload an Image')

    # Upload Image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=None)

        preprocessed_images = process_image(image)
        st.session_state["preprocessed_images"] = preprocessed_images
        st.success("Image preprocessed and ready for prediction.")

    else:
        st.info("Please upload an image to begin detection.")

with tab2:
    st.header("Step 2: Choose a Model and Detect")

    model_options = ["Logistic Regression", "CNN"]

    if "preprocessed_images" not in st.session_state:
         st.warning("⚠️ Please upload and preprocess an image in Tab1 first.")
    else:
        selected_model = st.selectbox("Select a model:", model_options)
        preprocessed_images = st.session_state["preprocessed_images"]

        if st.button("🧠 Detect AI Art"):
            if selected_model == "CNN":
                model = CNN()
                prediction = model.predict_proba(preprocessed_images['float32'])
                ai_probability = float(prediction[0][0])
            else:
                try:
                    # Pass the uint8 array directly
                    result = predict_single_image(preprocessed_images['uint8'])
                    ai_probability = result['probability']
                    prediction_label = result['prediction']
                    
                    st.subheader(f"🧾 Result for: **{selected_model}**")
                    st.write(f"🎯 This ART WORK is **{int(ai_probability * 100)}%** likely to be **AI-generated**.")
                    st.write(f"Prediction: **{prediction_label}**")
                    st.progress(min(max(ai_probability, 0.0), 1.0))
                    
                except ValueError as e:
                    st.error(str(e))
                    st.stop()

                with st.expander("ℹ️ How this result is calculated"):
                    st.markdown("""
                    This value is based on the model's predicted probability for the AI class.
                    
                    If the model assigns a probability greater than or equal to a certain threshold (e.g., 0.5), 
                    the image is classified as AI-generated.

                    Here, we visualize the raw probability score directly.
                    """)

                # show the evaluation result for the model
                metrics = get_performance_metrics()
                model_metrics = metrics["logistic_regression"] if selected_model == "Logistic Regression" else metrics["cnn"]

                st.write("### Model Metrics from Evaluation Function")
                st.write("Accuracy:", f"{model_metrics['accuracy']:.4f}")
                st.write("Precision:", f"{model_metrics['precision']:.4f}")
                st.write("Recall:", f"{model_metrics['recall']:.4f}")
                st.write("F1 Score:", f"{model_metrics['f1_score']:.4f}")
                st.write("Mean Squared Error (MSE):", f"{model_metrics['mse']:.4f}")
       
