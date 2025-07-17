# IMDB Movie Review Sentiment Analysis
This project implements a sentiment analysis model to classify IMDB movie reviews as positive or negative using a Simple Recurrent Neural Network (RNN) built with TensorFlow and Keras. The model is trained on the IMDB dataset and deployed as a web application using Streamlit for user interaction.
# Project Overview
## The project consists of:

A Jupyter notebook (simplernn.ipynb) for training and evaluating the Simple RNN model.
A Streamlit application (main.py) for classifying user-provided movie reviews.
A pre-trained model file (simple_rnn_imdb.h5) for sentiment prediction.
A requirements file (requirements.txt) listing the necessary Python libraries.

The model processes movie reviews by converting them into numerical sequences, padding them to a fixed length, and predicting sentiment using a trained Simple RNN. The Streamlit app allows users to input a movie review and receive a sentiment prediction (Positive or Negative) along with a confidence score.
## Files in the Repository

simplernn.ipynb: Jupyter notebook containing the code to load the IMDB dataset, preprocess the data, train the Simple RNN model, and save the trained model as simple_rnn_imdb.h5.
main.py: Python script for the Streamlit web application that loads the pre-trained model and allows users to input movie reviews for sentiment classification.
simple_rnn_imdb.h5: Pre-trained Simple RNN model file used for sentiment prediction in the Streamlit app.
requirements.txt: Lists the required Python libraries and their versions for running the project.

## Prerequisites
To run this project, you need:

Python 3.11 or compatible version
A virtual environment (recommended)
The libraries listed in requirements.txt

## Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


## Create and Activate a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


## Install Dependencies:
pip install -r requirements.txt


Download the Pre-trained Model:Ensure the simple_rnn_imdb.h5 file is present in the project directory. This file is required for the Streamlit app to make predictions.


## Running the Streamlit App
To launch the Streamlit application:
streamlit run main.py


Open the provided URL (usually http://localhost:8501) in your web browser.
Enter a movie review in the text area.
Click the "Classify" button to see the predicted sentiment (Positive or Negative) and the prediction score.

Training the Model
To retrain the model, open simplernn.ipynb in Jupyter Notebook or JupyterLab:

Install Jupyter if not already installed:pip install jupyter


Launch Jupyter Notebook:jupyter notebook


Open simplernn.ipynb and run all cells to preprocess the IMDB dataset, train the Simple RNN model, and save it as simple_rnn_imdb.h5.

Model Details

Dataset: IMDB movie review dataset from TensorFlow/Keras, with a vocabulary size of 10,000 words.
Preprocessing: Reviews are converted to numerical sequences, padded/truncated to a length of 500 words.
Model Architecture:
Embedding layer (10,000 vocab size, 128-dimensional embeddings)
Simple RNN layer (128 units, ReLU activation)
Dense layer (1 unit, sigmoid activation for binary classification)


Training:
Optimizer: Adam
Loss: Binary Crossentropy
Metrics: Accuracy
Early Stopping: Monitors validation loss with a patience of 5 epochs


Output: The model outputs a score between 0 and 1, where >0.5 indicates a positive sentiment and ≤0.5 indicates a negative sentiment.

Example Usage

Input a Review:

Example: "This movie was absolutely fantastic! The acting was superb, and the plot kept me engaged throughout."
Output: Sentiment: Positive, Prediction Score: ~0.85


Input a Review:

Example: "The movie was boring and poorly acted. I wouldn't recommend it."
Output: Sentiment: Negative, Prediction Score: ~0.32



Notes

The simple_rnn_imdb.h5 file is essential for the Streamlit app. Ensure it is not deleted or moved.
The model may not handle very short reviews or reviews with out-of-vocabulary words optimally due to the preprocessing steps.
For better performance, consider experimenting with LSTM or GRU layers instead of Simple RNN, or fine-tuning hyperparameters.

Requirements
The required libraries are listed in requirements.txt:
tensorflow==2.15.0
pandas
numpy
scikit-learn
tensorboard
matplotlib
streamlit
scikeras

Contributing
Feel free to fork this repository, create a feature branch, and submit a pull request with improvements or bug fixes.
License
This project is licensed under the MIT License. See the LICENSE file for details.
