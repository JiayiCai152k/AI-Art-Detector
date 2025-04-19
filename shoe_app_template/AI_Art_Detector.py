import streamlit as st
import numpy as np
from PIL import Image


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

    # Sample confusion matrices (TN, FP, FN, TP)
    sample_conf_matrices = {
        "Logistic Regression": np.array([[33, 7], [5, 41]]),
        "CNN": np.array([[28, 6], [4, 50]]),
    }

    conf_matrix = sample_conf_matrices[selected_model]
    true_neg, false_pos, false_neg, true_pos = conf_matrix.ravel()

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)

    # Display metrics
    st.write("### Model Metrics")
    st.write("**Precision:** %.2f" % precision)
    st.write("**Recall:** %.2f" % recall)

    st.write('There are **{}** false positives'.format(false_pos))
    st.write('There are **{}** false negatives'.format(false_neg))
    st.write('There are **{}** true positives'.format(true_pos))
    st.write('There are **{}** true negatives'.format(true_neg))