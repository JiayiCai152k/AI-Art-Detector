# AI Art Detector: Classifying AI-Generated vs. Human-Created Ukiyo-e Art

This project implements a machine learning system to distinguish between AI-generated and human-created Ukiyo-e artwork using both Logistic Regression and CNN approaches.

## Authors

- [Shiwen(Lareina) Yang](sy796@cornell.edu)
- [Jiayi Cai](jc3669@cornell.edu)
- [Zach Zhong](zz857@cornell.edu)

## Repository Structure

```
AI-Art-Detector/
├── data/                      # Dataset directory
│   ├── Human_Ukiyo_e/        # Original human-created Ukiyo-e images
│   ├── AI_SD_ukiyo-e/        # AI-generated Ukiyo-e images using Stable Diffusion
│   ├── Human_center_crop/    # Preprocessed human images (center cropped)
│   └── ADDITIONAL_AI_ART/    # Additional AI art samples
├── model/                    # Model implementations
├── data_preprocessing/       # Data preprocessing scripts
├── outputs/                  # Model outputs and evaluation results
├── AI_Art_Detector.py       # Main Streamlit application
└── requirements.txt         # Python dependencies
```

## Dataset

The dataset consists of two main categories:

1. Human-created Ukiyo-e artwork: Traditional Japanese woodblock prints
2. AI-generated Ukiyo-e artwork: Images created using Stable Diffusion

The data is preprocessed and stored in various formats:

- Original images
- Center-cropped versions for CNN
- Standardized versions for consistent input

## Pre-trained Models

We provide pre-trained weights for quick loading:

- CNN models with different learning rates and batch sizes
- Logistic regression model weights

All weights are saved in the root directory:

- `cnn_weights_epoch_5_*.pth`: Various CNN model weights
- `logistic_model_weights.npz`: Logistic regression weights

## Setup and Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd AI-Art-Detector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the Streamlit application:

```bash
streamlit run AI_Art_Detector.py
```

2. Use the application:
   - Upload an image in the first tab
   - Choose between Logistic Regression or CNN model
   - Click "Detect AI Art" to get the prediction

## Features

- Interactive web interface using Streamlit
- Support for both Logistic Regression and CNN models
- Real-time image processing and prediction
- Detailed model performance metrics
- Visualization of prediction probabilities
- Support for various image formats (jpg, jpeg, png)

## Model Performance

Both models provide the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Mean Squared Error (MSE)

For the Logistic Regression model, feature importance is also displayed.

## Notes

- The application uses MPS (Metal Performance Shaders) if available on macOS, otherwise falls back to CPU
- Images are automatically preprocessed (center-cropped, standardized) before prediction
- Model weights are loaded automatically from the provided pre-trained files
