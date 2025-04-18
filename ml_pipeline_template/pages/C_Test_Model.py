import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <project title>")

#############################################

st.title('Test Model')

#############################################

df = None
df = fetch_dataset()

if df is not None:
    st.markdown("### Get Performance Metrics")
    metric_options = ['placeholder']

    model_options = ['placeholder']

    trained_models = ['placeholder']

    # Select a trained classification model for evaluation
    model_select = st.multiselect(
        label='Select trained models for evaluation',
        options=trained_models
    )

    if (model_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(model_select))

        eval_button = st.button('Evaluate your selected classification models')

        if eval_button:
            st.session_state['eval_button_clicked'] = eval_button

        if 'eval_button_clicked' in st.session_state and st.session_state['eval_button_clicked']:
            st.markdown('### Review Model Performance')

            review_options = ['plot', 'metrics']

            review_plot = st.multiselect(
                label='Select plot option(s)',
                options=review_options
            )

            if 'plot' in review_plot:
                pass

            if 'metrics' in review_plot:
                pass

    # Select a model to deploy from the trained models
    st.markdown("### Choose your Deployment Model")
    model_select = st.selectbox(
        label='Select the model you want to deploy',
        options=trained_models,
    )

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        if('deploy_model' in st.session_state):
            st.session_state['deploy_model'] = st.session_state[model_select]

    st.write('Continue to Deploy Model')
