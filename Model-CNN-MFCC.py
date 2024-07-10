from google.colab import drive
import zipfile
import os

drive.mount('/content/drive')

# Mengatur jalur dataset
dataset_path = "/content/drive/MyDrive/PPDM/genres_original.zip"
dataset_dir = 'ckplus/genres_original'

# Ekstrak file zip ke direktori 'ckplus'
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall('ckplus')

# Daftar genre yang ingin diklasifikasikan
class_labels = ['classical', 'jazz', 'pop']

import numpy as np
import librosa
import os

def preprocess_audio(file_path, sample_rate=22050, frame_size=2048, hop_length=512, max_length=5):
    try:
        # Load the audio file
        signal, sr = librosa.load(file_path, sr=sample_rate)

        # Ensure the signal length is consistent
        if len(signal) < sample_rate * max_length:
            # Pad the signal
            pad_width = sample_rate * max_length - len(signal)
            signal = np.pad(signal, (0, pad_width), mode='constant')
        else:
            # Truncate the signal
            signal = signal[:sample_rate * max_length]

        # Normalize the signal
        signal = librosa.util.normalize(signal)

        # Windowing and framing (STFT)
        stft = librosa.stft(signal, n_fft=frame_size, hop_length=hop_length)

        # Compute the magnitude of the STFT
        magnitude = np.abs(stft)

        # Compute the power of the STFT
        power = magnitude**2

        return power, sr, signal
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None

def load_data(data_dir, class_labels, sample_rate=22050, frame_size=2048, hop_length=512, max_length=5):
    data = []
    labels = []
    signals = []
    for label in class_labels:
        genre_dir = os.path.join(data_dir, label)
        for filename in os.listdir(genre_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(genre_dir, filename)
                power, sr, signal = preprocess_audio(file_path, sample_rate, frame_size, hop_length, max_length)
                if power is not None:
                    data.append(power)
                    labels.append(label)
                    signals.append(signal)
    return np.array(data), np.array(labels), signals

# Load data and labels
data, labels, signals = load_data(dataset_dir, class_labels)

import matplotlib.pyplot as plt

# Pilih satu contoh sinyal untuk ditampilkan sebelum dan sesudah preprocessing
example_index = 1
example_signal = signals[example_index]

# Plot sinyal sebelum preprocessing
plt.figure(figsize=(14, 5))
plt.subplot(2, 1, 1)
plt.title('Original Signal')
plt.plot(example_signal)
plt.xlabel('Sample')
plt.ylabel('Amplitude')

# Proses ulang sinyal untuk mendapatkan sinyal setelah preprocessing
example_file_path = os.path.join(dataset_dir, class_labels[example_index], os.listdir(os.path.join(dataset_dir, class_labels[example_index]))[1])
power, sr, processed_signal = preprocess_audio(example_file_path)

# Plot sinyal setelah preprocessing
plt.subplot(2, 1, 2)
plt.title('Preprocessed Signal')
plt.plot(processed_signal)
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

# Assuming 'labels' contains the class labels after loading the data

# Calculate the count of each class
unique_labels, label_counts = np.unique(labels, return_counts=True)

# Plotting a pie chart
plt.figure(figsize=(8, 6))
patches, texts, autotexts = plt.pie(label_counts, labels=unique_labels, autopct='%1.1f%%', startangle=140)
plt.title('Data Distribution by Class')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Adding count labels
count_labels = [f'{label}: {count}' for label, count in zip(unique_labels, label_counts)]
plt.legend(patches, count_labels, loc="best")

plt.show()

def extract_features(data, sr=22050, n_mfcc=13):
    features = []
    for power in data:
        # Compute MFCCs from the power of the STFT
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(power), sr=sr, n_mfcc=n_mfcc)
        features.append(np.expand_dims(mfccs, axis=-1))
    return np.array(features)

# Extract MFCC features
features = extract_features(data)

import random
import matplotlib.pyplot as plt

# Select a random index to display its MFCC dimensions and values
random_index = random.randint(0, len(features) - 1)
mfcc_features = features[random_index]

# Print the dimensions and explanation
if mfcc_features.shape[0] == 13:
    mfcc_time_steps = mfcc_features.shape[1]
    mfcc_channels = mfcc_features.shape[2]
    total_features = mfcc_features.shape[0] * mfcc_features.shape[1] * mfcc_features.shape[2]

    print(f"MFCC dimensions for data at index {random_index}: {mfcc_features.shape}")
    print(f"- {mfcc_features.shape[0]} MFCC coefficients")
    print(f"- {mfcc_features.shape[1]} time frames")
    print(f"- {mfcc_features.shape[2]} channel")
    print(f"Total features for this document: {total_features}")

    # Plot the MFCCs
    plt.figure(figsize=(10, 4))
    plt.imshow(mfcc_features.squeeze(), aspect='auto', origin='lower')
    plt.title(f"MFCC for sample at index {random_index}")
    plt.xlabel('Time Frames')
    plt.ylabel('MFCC Coefficients')
    plt.colorbar()
    plt.show()

    # Print the MFCC values
    print("MFCC values:")
    for i in range(mfcc_features.shape[0]):
        print(f"MFCC {i+1}: {mfcc_features[i]}")
else:
    print(f"Expected 13 MFCCs, but got {mfcc_features.shape[0]} MFCCs")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels_categorical, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Hyperparameters
def create_model(dropout_rate=0.5, learning_rate=0.001, conv_filters_1=32, conv_filters_2=64):
    model = Sequential()
    model.add(Conv2D(conv_filters_1, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(conv_filters_2, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(class_labels), activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Define hyperparameters
dropout_rate = 0.5
learning_rate = 0.001
conv_filters_1 = 32
conv_filters_2 = 64

X_train_cnn = np.expand_dims(X_train, axis=3)
X_test_cnn = np.expand_dims(X_test, axis=3)

# Create and train the model
model = create_model(dropout_rate=dropout_rate, learning_rate=learning_rate, conv_filters_1=conv_filters_1, conv_filters_2=conv_filters_2)
history = model.fit(X_train_cnn, y_train, epochs=250, batch_size=32, validation_split=1/12.)

# Evaluate the model
train_acc = np.mean(history.history['accuracy'])
print(f"Average training accuracy: {train_acc}")

# Simpan model ke file
model.save('model_trained.h5')
print("Model berhasil disimpan.")

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test_cnn, y_test)
print(f"Test accuracy: {test_acc}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()

# Predict the test set results
y_pred = model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plotting confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
