from flask import Flask, render_template_string, request, redirect, url_for
import os
import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from werkzeug.utils import secure_filename
import librosa

app = Flask(__name__)

# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ----------------------- Model Code -----------------------
random.seed(120)
np.random.seed(120)
torch.manual_seed(120)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

file_path = "C:\\Users\\gumpe\\Downloads\\Project School\\project ANN\\combined_music_dataset.csv"
music_dataset = pd.read_csv(file_path)
X = music_dataset.drop(columns=['filename', 'genre', 'duration(in sec)'])
y = music_dataset['genre']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train.factorize()[0])
y_test_tensor = torch.LongTensor(y_test.factorize()[0])

class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

input_size = X_train.shape[1]
num_classes = len(y.unique())
model = MusicGenreClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)

conf_matrix = confusion_matrix(y_test_tensor, predicted)
class_report = classification_report(y_test_tensor, predicted, target_names=y.unique())



# ----------------------- Routes -----------------------

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Automatic Music Classification</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: #000;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .book-container {
                position: relative;
                width: 60%;
                height: 60%;
                display: flex;
                justify-content: center;
                align-items: center;
                perspective: 1200px;
            }
            .book-image {
                width: 100%;
                height: 100%;
                background: url('/static/music_image.jpg') no-repeat center center;
                background-size: cover;
                position: absolute;
                z-index: 2;
            }
            .book-half {
                position: absolute;
                width: 50%;
                height: 100%;
                background: url('/static/music_image.jpg') no-repeat center center;
                background-size: cover;
                transition: transform 2s ease-in-out;
            }
            .left-half { left: 0; background-position: left center; transform-origin: left center; }
            .right-half { right: 0; background-position: right center; transform-origin: right center; }
            .book-container.open .book-image { opacity: 0; }
            .book-container.open .left-half { transform: rotateY(-90deg); }
            .book-container.open .right-half { transform: rotateY(90deg); }
            .welcome-text {
                position: absolute;
                font-family: 'Arial', sans-serif;
                font-size: 2.5em;
                color: white;
                text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.8);
                text-align: center;
                opacity: 0;
                transition: opacity 2s ease-in-out 2s;
                z-index: 1;
            }
            .book-container.open .welcome-text { opacity: 1; }
            .overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                color: white;
                font-size: 1.5em;
                font-family: Arial, sans-serif;
                opacity: 0;
                transition: opacity 1s ease-in-out;
            }
            .overlay.visible { opacity: 1; }
        </style>
    </head>
    <body>
        <div class="book-container" id="book">
            <div class="book-image"></div>
            <div class="book-half left-half"></div>
            <div class="book-half right-half"></div>
            <div class="welcome-text">Welcome to the <br> Automatic Music Classification</div>
        </div>
        <div class="overlay" id="overlay">Press any key to continue...</div>

        <script>
            window.onload = function() {
                const book = document.getElementById('book');
                const overlay = document.getElementById('overlay');

                // Start the book animation without any speech
                setTimeout(() => {
                    book.classList.add('open');
                }, 3000);  // Slight delay for animation

                // Show the overlay with key press hint after 10 seconds
                setTimeout(() => {
                    overlay.classList.add('visible');
                }, 10000);
            };

            document.addEventListener('keydown', function() {
                window.location.href = "/welcome";
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/welcome')
def welcome():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evaluation</title>
            <link href="https://fonts.googleapis.com/css2?family=Dancing+Script:wght@700&family=Pacifico&display=swap" rel="stylesheet">
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #2a3d66;
                    color: #ffffff;
                    text-align: center;
                    padding: 20px;
                    height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 0;
                }
                h1 {
                    color: #ffffff;
                }
                .evaluation-text {
                    font-family: 'Dancing Script', cursive;
                    font-size: 4em;
                    color: white;
                    opacity: 0;
                    transition: opacity 2s ease-in-out;
                    padding-top: 50px;
                    text-align: center;
                }
                .file-upload-form {
                    display: none;
                    margin-top: 20px;
                    text-align: center;
                }
                .file-upload-form.show {
                    display: block;
                }
            </style>
        </head>
        <body>
            <div class="evaluation-text" id="evaluationText">Evaluation</div>
            <div class="file-upload-form" id="fileUploadForm">
                <h1>Upload an Audio File for Prediction</h1>
                <form method="POST" action="/predict" enctype="multipart/form-data">
                    <input type="file" name="file" accept="audio/*" required><br>
                    <button type="submit">Upload and Predict Genre</button>
                </form>
            </div>

            <script>
                window.onload = function() {
                    const evaluationText = document.getElementById('evaluationText');
                    const fileUploadForm = document.getElementById('fileUploadForm');

                    // After 6 seconds, show file upload form and hide Evaluation text
                    setTimeout(() => {
                        evaluationText.style.opacity = 0;
                        fileUploadForm.style.display = 'block';
                    }, 1000);
                };
            </script>
        </body>
        </html>
    ''')

@app.route('/thankyou')
def thank_you():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Thank You</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #2a3d66;
                color: #ffffff;
                text-align: center;
                padding: 50px;
            }
            h1 {
                font-size: 36px;
                margin-bottom: 20px;
            }
            p {
                font-size: 18px;
            }
        </style>
    </head>
    <body>
        <h1>Thank You!</h1>
        <p>Press any key to close this page.</p>

        <script>
            document.addEventListener('keydown', function() {
                window.close();
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs = np.mean(mfccs, axis=1)
    mfccs_scaled = scaler.transform([mfccs])

    input_tensor = torch.FloatTensor(mfccs_scaled)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_class = torch.max(output, 1)

    genre = y_train.unique()[predicted_class.item()]

    # Map genre to images
    genre_image_map = {
        'classical': 'classical.jpg',
        'Jazz': 'jazz.jpg',
        'Pop': 'pop.jpg',
        'Rock': 'rock.jpg'
    }
    genre_image = genre_image_map.get(genre)



    return render_template_string(''' 
    <!DOCTYPE html>
    <html>
    <head>
        <title>Prediction Result</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #2a3d66;
                color: #ffffff;
                text-align: center;
                padding: 20px;
            }
            h1 {
                color: #ffffff;
                margin-bottom: 20px;
            }
            .genre-text {
                font-size: 24px;
                margin-bottom: 20px;
            }
            .genre-image {
                max-width: 300px;
                height: auto;
                border-radius: 10px;
                border: 2px solid #ffffff;
                margin-bottom: 20px;
            }
            .thank-you {
                margin-top: 30px;
                font-size: 18px;
                color: #f1f1f1;
            }
        </style>
    </head>
    <body>
        <h1>Predicted Genre</h1>
        <p class="genre-text">{{ genre }}</p>
        <img src="/static/{{ genre_image }}" alt="Genre Image" class="genre-image">
        <p class="thank-you">Press any key to continue...</p>

        <script>
            document.addEventListener('keydown', function() {
                window.location.href = "/thankyou";
            });
        </script>
    </body>
    </html>
    ''', genre=genre, genre_image=genre_image)
    
if __name__ == "__main__":
    app.run(debug=True)
    
