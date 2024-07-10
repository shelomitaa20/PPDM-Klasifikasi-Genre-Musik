import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import base64
import tempfile
import os
import pandas as pd

# Load model
model = load_model('model_trained (1).h5')
class_labels = ['classical', 'jazz', 'pop'] 

# Konfigurasi halaman
st.set_page_config(page_title="Music Genre Classification", layout='centered', initial_sidebar_state='expanded', menu_items=None)

# Styling
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        body {
            background-color: #f0f8ff;
            color: #004080;
        }
        .css-1d391kg {
            background-color: #004080;
            color: white;
        }
        .st-emotion-cache-1ab9dzl {
            margin: 50px 0px;
        }
        .st-emotion-cache-17i9sy0 {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
        }
        .css-2trqyj {
            border: 1px solid #004080;
            border-radius: 10px;
        }
        table {
            width: 100%;
            background-color: #f0f8ff;
            color: #004080;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            border: 1px solid #004080;
            text-align: center;
        }
        th {
            background-color: #004080;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #e6f2ff;
        }
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

def get_audio_html(audio_path, alt_text="Audio", width=None):
    audio_file = open(audio_path, 'rb')
    audio_bytes = audio_file.read()
    audio_str = base64.b64encode(audio_bytes).decode()
    width_str = f' width="{width}"' if width else ""
    audio_html = f'<audio controls src="data:audio/wav;base64,{audio_str}"{width_str}></audio>'
    return audio_html

# Navbar
st.markdown(f"""
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
<script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
<nav class="navbar fixed-top navbar-expand-lg navbar-light" style="background-color: #004080;">
    <div class="container">
        <span class="navbar-brand mb-0 h1 text-white">Kelompok B1</span>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbar-list-2" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbar-list-2">
            <ul class="navbar-nav ml-auto mt-2 mt-lg-0">
                <li class="nav-item">
                    <a class="nav-link text-white" href="" target="_blank">About Us</a>
                </li>
                <li class="nav-item">
                    <a class="nav-link text-white" href="" target="_blank">FAQ</a>
                </li>
                <li class="nav-item">
                    <button type="button" class="btn btn-light btn-md py-2 px-3" style="width: 166px">Sign In</button>
                </li>
            </ul>
        </div>
    </div>
</nav>
""", unsafe_allow_html=True)

def extract_mfcc_features(audio_path, n_mfcc=13, max_pad_len=216):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    return mfccs

uploaded_file = st.file_uploader('Upload an Audio File', type=['wav', 'mp3'], accept_multiple_files=True)

# Initialize a session state to store uploaded audios
if 'uploaded_audios' not in st.session_state:
    st.session_state.uploaded_audios = []
if 'results' not in st.session_state:
    st.session_state.results = []

if uploaded_file:
    for file in uploaded_file:
        st.session_state.uploaded_audios.append(file)
    # Conditionally display the submit button
    if st.session_state.uploaded_audios:
        if st.button("Submit"):
            if st.session_state.uploaded_audios:
                if uploaded_file is not None:
                    for uploaded_file in uploaded_file:
                        # Save the uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_file.write(uploaded_file.read())
                            temp_file_path = temp_file.name

                        # Extract MFCC features
                        mfcc_features = extract_mfcc_features(temp_file_path)
                        mfcc_features = mfcc_features.reshape(1, mfcc_features.shape[0], mfcc_features.shape[1], 1)  # Reshape to match model input

                        # Predict the class
                        prediction = model.predict(mfcc_features)
                        predicted_label = class_labels[np.argmax(prediction)]

                        # Display the audio player
                        audio_html = get_audio_html(temp_file_path)
                        st.markdown(audio_html, unsafe_allow_html=True)

                        # Append the result to session state
                        st.session_state.results.append({
                            "file_name": uploaded_file.name,
                            "audio": temp_file_path,
                            "label": predicted_label
                        })
    else:
        st.write('No audios uploaded yet.')

if st.session_state.results:
    with st.container():
        st.title('Classification')
        df = pd.DataFrame({
            "File Name": [result["file_name"] for result in st.session_state.results],
            "Audio": [get_audio_html(result["audio"]) for result in st.session_state.results],
            "Prediction": [result['label'] for result in st.session_state.results],
        })

        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

st.markdown("""<script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>""", unsafe_allow_html=True)