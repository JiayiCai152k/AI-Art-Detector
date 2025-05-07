import streamlit as st
import numpy as np
from PIL import Image
from model.eval import get_performance_metrics
from model.logistic import LogisticRegression
from model.cnn import CNN

def process_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  
    st.write("Image resized to", img_array.shape)
    return img_array
    

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
                st.progress(min(max(ai_probability, 0.0), 1.0))  # Ensure between 0 and 1

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
                st.write("Accuracy:", f"{model_metrics['accuracy']:.2f}")
                st.write("Precision:", f"{model_metrics['precision']:.2f}")
                st.write("Recall:", f"{model_metrics['recall']:.2f}")
                st.write("F1 Score:", f"{model_metrics['f1_score']:.2f}")
                st.write("Mean Squared Error (MSE):", f"{model_metrics['mse']:.2f}")
       
