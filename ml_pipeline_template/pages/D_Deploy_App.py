import streamlit as st
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - <project title>")

#############################################

st.title('Deploy Application')

#############################################

df = None
if 'data' in st.session_state:
    df = st.session_state['data']
else:
    st.write(
        '### The <project> Application is under construction. Coming to you soon.')

# Deploy App
if df is not None:
    st.markdown('### <Deployment app name>')

    st.markdown('#### Some descriptions about the deployment app')
