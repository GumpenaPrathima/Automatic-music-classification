import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Set random seeds for reproducibility
random.seed(120)
np.random.seed(120)
torch.manual_seed(120)
torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False  

# Load the dataset
file_path = "c:\\Users\\gumpe\\Downloads\\Project School\\combined_music_dataset.csv"
music_dataset = pd.read_csv(file_path)

# Step 1: Separate features and labels
X = music_dataset.drop(columns=['filename', 'genre', 'duration(in sec)'])  # Features (MFCCs)
y = music_dataset['genre']  # Labels (genres)

# Step 2: Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and test sets (70% training, 30% testing) with stratified sampling
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_index, test_index in sss.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
y_train_tensor = torch.LongTensor(y_train.factorize()[0])  # Convert categories to integers
y_test_tensor = torch.LongTensor(y_test.factorize()[0])

# Step 4: Define the neural network model
class MusicGenreClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MusicGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)  # Increased neurons in the first layer
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_classes)  # Output layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Added dropout for regularization

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.fc5(x)  # No activation here, softmax in loss function
        return x

# Step 5: Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Number of features
num_classes = len(y.unique())   # Number of unique genres
model = MusicGenreClassifier(input_size, num_classes)

criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate

# Step 6: Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:  # Print every 10 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

## Save the model's state dictionary
torch.save(model.state_dict(), "music_genre_classifier.pth")
print("Model saved successfully!")


# Step 7: Evaluate the model
model.eval()  # Set the model to evaluation mode (turns off dropout)
with torch.no_grad():
    y_pred = model(X_test_tensor)
    _, predicted = torch.max(y_pred, 1)

# Step 8: Print evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test_tensor, predicted))

print("\nClassification Report:")
print(classification_report(y_test_tensor, predicted, target_names=y.unique()))