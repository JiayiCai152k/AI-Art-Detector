import streamlit as st
import numpy as np
from PIL import Image
from model.eval import get_performance_metrics

def apply_threshold(probabilities, threshold):
    # +1 if >= threshold and -1 otherwise.
    return np.array([1 if p[1] >= threshold else 0 for p in probabilities])
    
tab1, tab2 = st.tabs(["AI-Art-Detector", "Detection Analysis"])

with tab1:
    st.header("AI Art Detection Tool") 
    st.markdown('### Import an Image')

    # Upload Image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("Please upload an image to begin detection.")

with tab2:
    st.header("Detection System Analysis")


    model_options = ["Logistic Regression", "CNN"]
    selected_model = st.selectbox("Select a model:", model_options)

    st.write('### Detection Evaluation Performance')

    st.markdown(f"#### Showing performance for: **{selected_model}**")

    # result
    ai_probability = 0.40

    st.subheader("Results")
    st.write(f"This ART WORK is **{int(ai_probability * 100)}%** likely to be **AI Generated!!!!**")
    st.progress(ai_probability)

    with st.expander("ℹ️ How this result is calculated"):
        st.markdown("""
        This value is based on the model's predicted probability for the AI class.
        
        If the model assigns a probability greater than or equal to a certain threshold (e.g., 0.5), 
        the image is classified as AI-generated.

        Here, we visualize the raw probability score directly.
        """)

    # show the evaluation result for the model
    metrics = get_performance_metrics()
    if selected_model == "Logistic Regression":
        model_metrics = metrics["logistic_regression"]
    else:
        model_metrics = metrics["cnn"]

    st.write("### Model Metrics from Evaluation Function")
    st.write("Accuracy:", f"{model_metrics['accuracy']:.2f}")
    st.write("Precision:", f"{model_metrics['precision']:.2f}")
    st.write("Recall:", f"{model_metrics['recall']:.2f}")
    st.write("F1 Score:", f"{model_metrics['f1_score']:.2f}")
    st.write("Mean Squared Error (MSE):", f"{model_metrics['mse']:.2f}")
