import streamlit as st                  # pip install streamlit
from helper_functions import fetch_dataset

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <project title>")

#############################################

st.title('Train Model')

#############################################
df = None
df = fetch_dataset()


def inspect_coefficients(models):
    pass


if df is not None:
    # Display dataframe as table
    st.dataframe(df)

    # Select variable to predict
    st.markdown('### Select variable to predict')
    feature_predict_select = st.selectbox(
        label='Select variable to predict',
        options=df.columns,
        key='feature_selectbox',
    )

    # Select input features
    st.markdown('### Select input features')
    feature_input_select = st.selectbox(
        label='Select features for classification input',
        options=df.columns,
        key='feature_select'
    )

    st.write('You selected input {} and output {}'.format(
        feature_input_select, feature_predict_select))

    # Split dataset
    st.markdown('### Split dataset into Train/Validation/Test sets')
    st.markdown(
        '#### Enter the percentage of validation/test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)

    # Train models
    st.markdown('### Train models')
    model_options = ['Placehoder']

    # Collect ML Models of interests
    model_select = st.multiselect(
        label='Select regression model for prediction',
        options=model_options,
    )
    st.write('You selected the follow models: {}'.format(model_select))

    if (model_options[0] in model_select):
        st.markdown('#### ' + model_options[0])

        param_col1, param_col2 = st.columns(2)
        with (param_col1):
            param1_options = []
            param1_select = st.selectbox(
                label='Select param1',
                options=param1_options,
                key='param1_select'
            )
            st.write('You select the following <param1>: {}'.format(param1_select))

            param2_options = []
            param2_select = st.selectbox(
                label='Select param2',
                options=param2_options,
                key='param2_select'
            )
            st.write('You select the following <param2>: {}'.format(param2_select))

        with (param_col2):
            param3_options = []
            param3_select = st.selectbox(
                label='Select param3',
                options=param3_options,
                key='param3_select'
            )
            st.write('You select the following <param3>: {}'.format(param3_select))

            param4_options = []
            param4_select = st.selectbox(
                label='Select param4',
                options=param4_options,
                key='param4_select'
            )
            st.write('You select the following <param4>: {}'.format(param4_select))

        model_params = {
            'param1': param1_select,
            'param2': param2_select,
            'param3': param3_select,
            'param4': param4_select
        }

        if st.button('Placeholder Model'):
            # train this model
            pass

        if model_options[0] not in st.session_state:
            st.write('Placeholder Model is untrained')
        else:
            st.write('Placeholder Model trained')

    # Inspect coefficients
    st.markdown('### Inspect model coefficients')

    inspect_models = st.multiselect(
        label='Select models to inspect coefficients',
        options=model_select,
        key='inspect_multiselect'
    )

    models = {}
    for model_name in inspect_models:
        if (model_name in st.session_state):
            models[model_name] = st.session_state[model_name]

    coefficients = inspect_coefficients(models)

    st.write('Continue to Test Model')
