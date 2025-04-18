import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



st.set_page_config( 
    page_title="NYC Traffic Accident",  # <- Add a title
    page_icon=":red_car:", # <- Change this to an emoji or something from https://fontawesome.com/icons?d=gallery&p=2&q=car&m=free
    layout="wide", # <- Specify a theme or remove this parameter to use your app's default theme
    initial_sidebar_state="expanded", # <- Specify which sidebar state to display by default
    
)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("NYC Traffic Accident") # <- Add a title
with col2:
    st.image("nypd-patch.png", width=100) # <- Add an image

st.markdown("""
This application is a Streamlit dashboard that can be used to analyze traffic accidents in NYC 🗽💥🚗. 
The data includes information about accident location, date, time, contributing factors, and more.
Use the tabs to navigate between different sections of the app.
""")
def load_data(): # load data here
    df = pd.read_csv('NYC Accidents 2020.csv')
    return df

def explore_data(df, subheader): # here you can add more features to explore
    st.subheader(subheader)
    
    st.write(df)

def describe_features(df): # add describe features here
    st.subheader('Features Description') 
    features_description = pd.DataFrame({
        'Feature': df.columns,
        'Description': [
            'Crash date',
            'Crash time',
            'Borough',
            'ZIP Code',
            'Latitude',
            'Longitude',
            'Location',
            'On street name',
            'Cross street name',
            'Off street name',
            'Number of persons injured',
            'Number of persons killed',
            'Number of pedestrians injured',
            'Number of pedestrians killed',
            'Number of cyclist injured',
            'Number of cyclist killed',
            'Number of motorist injured',
            'Number of motorist killed',
            'Contributing factor vehicle 1',
            'Contributing factor vehicle 2',
            'Contributing factor vehicle 3',
            'Contributing factor vehicle 4',
            'Contributing factor vehicle 5',
            'Collision ID',
            'Vehicle type code 1',
            'Vehicle type code 2',
            'Vehicle type code 3',
            'Vehicle type code 4',
            'Vehicle type code 5'
        ]
    })
    st.write(features_description)

def draw_map(df):
    st.subheader("Accident Locations")
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    st.map(df[["LATITUDE", "LONGITUDE"]])
# You can add more sections here



def preprocess_data(df): # add  preprocessing steps here
    st.header("Add some preprocessing steps here")


def train_model(): # add training steps here
    st.header("Add trainning steps here")

def predict(): # add prediction steps here
     st.header("Add prediction steps here")

def show_images(image_path,caption): # add image path and caption here
    st.image(image_path, caption=caption, width=300)

def show_images_in_columns(): 
    col1, col2 = st.columns(2)
    with col1: #
        image_path = 'GOHS logo transparent.png' # add image path here
        show_images(image_path,'GOHS logo transparent') # add caption here
    with col2:
        image_path = 'nypd-patch.png' # add image path here

        show_images(image_path,'nypd-patch') # add caption here

def main():
    df = load_data() # load data
    show_images_in_columns() # show images in columns
    # Create tab interface
    tabs= ["Show Dataset", "Show Features", "Feature Description",  "Show Map", "Preprocessing", "Training", "Prediction"] # add more tabs here
    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(tabs) 
    with tab1: # add code for each tab here
        explore_data(df,subheader='Show Dataset') 
    with tab2:
        explore_data(df.columns,subheader='Show Features')
        
    with tab3:
        
        describe_features(df)
    with tab4:
        draw_map(df)
    with tab5:
        preprocess_data(df)
    with tab6:
        train_model()
    with tab7: 
        predict() 
if __name__ == '__main__':
    main()
