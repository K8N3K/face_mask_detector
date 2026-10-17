# Face Mask Detector

A lightweight Python web app that detects whether a person is wearing a face mask in an uploaded image.

This project uses a TensorFlow/Keras image classification model and a Streamlit interface for easy image upload and prediction.

## Features

- Upload an image in JPG, JPEG, or PNG format
- Detects:
  - Not wearing a mask
  - Wearing a mask
- Displays the predicted class and confidence score
- Runs in a simple browser-based UI

## Project Structure

- `app.py` — Streamlit app and inference logic
- `face_mask_detector.keras` — trained Keras model
- `requirements.txt` — Python dependencies
- Sample test images in the repository root

## Model Details

The application loads a pre-trained Keras model and preprocesses uploaded images to a 128x128 RGB format before predicting the class.

Predicted labels:
- `Not wearing a mask`
- `Wearing a mask`

## Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
