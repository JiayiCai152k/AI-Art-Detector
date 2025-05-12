import streamlit as st
import numpy as np
from PIL import Image
import torch
from model.eval import get_performance_metrics
from model.logistic import (
    predict_single_image,
    output_model_results,
)
from model.cnn import CNN
from data_preprocessing.standarize import center_crop_image


def process_image(img, model_type="Logistic Regression", device=None):
    img = img.convert("RGB")

    # Create uint8 version for OpenCV operations (0-255)
    img_array_uint8 = np.array(img, dtype=np.uint8)

    # For CNN, we need to center crop and create a dataset
    if model_type == "CNN":
        cropped_img = center_crop_image(img)
        return {"uint8": img_array_uint8, "processed_image": cropped_img}
    else:
        img_array_float = img_array_uint8.astype(np.float32) / 255.0
        return {"uint8": img_array_uint8, "float32": img_array_float}


st.header("AI Art Detection Tool")

tab1, tab2 = st.tabs(["1️⃣ Upload Image", "2️⃣ Detect AI Art"])

with tab1:
    st.header("Step 1: Upload an Image")

    # Upload Image
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=None)

        # Store the original image for later use
        st.session_state["original_image"] = image
        st.success("Image uploaded successfully.")

    else:
        st.info("Please upload an image to begin detection.")

with tab2:
    st.header("Step 2: Choose a Model and Detect")

    model_options = ["Logistic Regression", "CNN"]

    if "original_image" not in st.session_state:
        st.warning("⚠️ Please upload an image in Tab1 first.")
    else:
        selected_model = st.selectbox("Select a model:", model_options)

        if st.button("🧠 Detect AI Art"):
            if selected_model == "CNN":
                try:
                    # Set up device
                    device = torch.device(
                        "mps" if torch.backends.mps.is_available() else "cpu"
                    )

                    # Initialize and move model to device
                    model = CNN(
                        load_weights_path="cnn_best_weights_epoch_5_lr_0.0001_bs_3.pth"
                    )
                    model = model.to(device)

                    # Process image with device information
                    preprocessed_images = process_image(
                        st.session_state["original_image"], selected_model, device
                    )

                    # Display original and processed images side by side
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Original Image:")
                        st.image(
                            st.session_state["original_image"], use_column_width=True
                        )
                        original_size = st.session_state["original_image"].size
                        st.write(f"Size: {original_size[0]}x{original_size[1]}")
                    with col2:
                        st.write("Processed Image (Center Cropped):")
                        st.image(
                            preprocessed_images["processed_image"],
                            use_column_width=True,
                        )
                        cropped_size = preprocessed_images["processed_image"].size
                        st.write(f"Size: {cropped_size[0]}x{cropped_size[1]}")

                    # Get prediction probabilities using the new method
                    probabilities = model.predict_single_image_proba(
                        preprocessed_images["processed_image"]
                    )
                    # probabilities is shape (1, 1), get the single value
                    ai_probability = float(probabilities[0, 0])

                    st.subheader(f"🧾 Result for: **{selected_model}**")
                    st.write(
                        f"🎯 This ART WORK is **{int(ai_probability * 100)}%** likely to be **AI-generated**."
                    )
                    st.write(
                        f"Prediction: **{'AI-generated' if ai_probability > 0.5 else 'Human-created'}**"
                    )
                    st.progress(min(max(ai_probability, 0.0), 1.0))

                except Exception as e:
                    st.error(f"An error occurred during CNN prediction: {str(e)}")
                    import traceback

                    st.write("Traceback:", traceback.format_exc())
            else:
                # Process image for Logistic Regression
                preprocessed_images = process_image(
                    st.session_state["original_image"], selected_model
                )
                try:
                    # Pass the uint8 array directly
                    result = predict_single_image(preprocessed_images["uint8"])
                    ai_probability = result["probability"]
                    prediction_label = result["prediction"]

                    st.subheader(f"🧾 Result for: **{selected_model}**")
                    st.write(
                        f"🎯 This ART WORK is **{int(ai_probability * 100)}%** likely to be **AI-generated**."
                    )
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
                metrics, importance = output_model_results()
                st.write("### Model Metrics from Evaluation Function")
                st.write("Accuracy:", f"{metrics['accuracy']:.4f}")
                st.write("Precision:", f"{metrics['precision']:.4f}")
                st.write("Recall:", f"{metrics['recall']:.4f}")
                st.write("F1 Score:", f"{metrics['f1_score']:.4f}")
                st.write("Mean Squared Error (MSE):", f"{metrics['mse']:.4f}")

                st.write("### Feature Importance")
                st.write(importance)
