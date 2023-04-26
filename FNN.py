import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from skorch import NeuralNetClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import folium
from tqdm import tqdm
data = pd.read_csv("San_Francisco.csv")
# Count the number of examples in each category
category_counts = data["Category"].value_counts()

# Print the counts for each category
print(category_counts)
# Load and preprocess data


data["Time"] = pd.to_datetime(data["Time"]).astype(int) / 10**9

encoder = LabelEncoder()
data["Category"] = encoder.fit_transform(data["Category"])
data["Part_of_Day"] = encoder.fit_transform(data["Part_of_Day"])
data["Day_of_Week"] = encoder.fit_transform(data["Day_of_Week"])

#Drop date
data.drop(["Date"], axis=1, inplace=True)

scaler = MinMaxScaler()
data[["Time", "Day_of_Week", "Part_of_Day", "Latitude", "Longitude"]] = scaler.fit_transform(
    data[["Time", "Day_of_Week", "Part_of_Day","Latitude", "Longitude"]]
)
# Prepare dataset and dataloader
class CrimeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError
        features = torch.tensor(
            self.data.loc[idx, ['Time','Day_of_Week','Part_of_Day','Latitude','Longitude']].values, dtype=torch.float
        )
        label = torch.tensor(self.data.loc[idx, 'Category'], dtype=torch.long)
        return features, label


dataset = CrimeDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Split dataset into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

train_dataset = CrimeDataset(train_data)
test_dataset = CrimeDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Define Feedforward Neural Network
class CrimeNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CrimeNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out


# Model parameters
input_size = 5
hidden_size = 64
num_classes = len(data["Category"].unique())

# Initialize model, loss function, and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CrimeNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5

for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}"
            )
# Test the model
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for features, labels in test_dataloader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


# Save the trained model
torch.save(model.state_dict(), "model.ckpt")
# Calculate accuracy, confusion matrix, and classification report
conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions)
accuracy = accuracy_score(all_labels, all_predictions)

print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
print("Accuracy:", accuracy)


from folium.plugins import MarkerCluster

# ...

# Create a map centered at the mean latitude and longitude
m = folium.Map(
    location=[test_data_inv["Latitude"].mean(), test_data_inv["Longitude"].mean()],
    zoom_start=12,
)

# Create a MarkerCluster
marker_cluster = MarkerCluster().add_to(m)

counter = 0
for idx, row in test_data_inv.iterrows():
    if row["Category"] == encoder.inverse_transform([all_predictions[counter]])[0]:
        marker_color = "green"
    else:
        marker_color = "red"
    folium.Marker(
        [row["Latitude"], row["Longitude"]],
        popup=row["Category"],
        icon=folium.Icon(color=marker_color),
    ).add_to(marker_cluster)
    counter += 1


# Save the map as an HTML file
m.save("crime_map.html")
