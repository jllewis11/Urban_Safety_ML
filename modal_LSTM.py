import time

import modal
import pandas as pd

stub = modal.Stub(
    "example-import-torch",
    image=modal.Image.debian_slim().pip_install(
        "pandas",
        "scikit-learn",
        "numpy",
        "folium",
<<<<<<< HEAD
        "torch", find_links="https://download.pytorch.org/whl/cu116",
=======
        "torch",
        find_links="https://download.pytorch.org/whl/cu116",
>>>>>>> 3ec5449 (Added lSTM)
    ),
)


<<<<<<< HEAD
<<<<<<< HEAD
@stub.function(gpu=modal.gpu.A10G(count=4), timeout=60*60*6)
=======
@stub.function(gpu=modal.gpu.T4(count=4), timeout=60 * 60 * 6)
>>>>>>> 3ec5449 (Added lSTM)
=======
@stub.function(gpu=modal.gpu.A10G(count=4), timeout=60 * 60 * 6)
>>>>>>> a9f5826 (Updated LSTM Accuracy still low)
def gpu_function(data):
    import subprocess

    import pandas as pd
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler, LabelEncoder
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import folium

    subprocess.run(["nvidia-smi"])
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    # Load and preprocess data
<<<<<<< HEAD
    
    data['Datetime'] = pd.to_datetime(data['Datetime']).astype(int) / 10**9
    scaler = MinMaxScaler()
    data[['Datetime', 'Latitude', 'Longitude']] = scaler.fit_transform(data[['Datetime', 'Latitude', 'Longitude']])
    encoder = LabelEncoder()
    data['Category'] = encoder.fit_transform(data['Category'])
=======

    # Column Date to Datetime
    data["Date"] = pd.to_datetime(data["Date"]).astype(int) / 10**9
    data["Time"] = pd.to_datetime(data["Time"], format="%H:%M").astype(int) / 10**9


    scaler = MinMaxScaler()
    data[["Date", "Time", "Latitude", "Longitude"]] = scaler.fit_transform(
        data[["Date", "Time", "Latitude", "Longitude"]]
    )
    encoder = LabelEncoder()
    data["Category"] = encoder.fit_transform(data["Category"])
>>>>>>> 3ec5449 (Added lSTM)

    # Prepare dataset and dataloader
    class CrimeDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
<<<<<<< HEAD
            features = torch.tensor(self.data.iloc[idx, [0, 2, 3]].values, dtype=torch.float)
            label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.long)
=======
            features = torch.tensor(
                self.data.iloc[idx, [0, 1, 3, 4]].values, dtype=torch.float
            )
            label = torch.tensor(self.data.iloc[idx, 2], dtype=torch.long)
>>>>>>> 3ec5449 (Added lSTM)
            return features, label

    dataset = CrimeDataset(data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Split dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Prepare dataset and dataloader for train and test sets
    train_dataset = CrimeDataset(train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = CrimeDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define LSTM model
    class CrimeLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(CrimeLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply hidden_size by 2 since it's a bidirectional LSTM

        def forward(self, x):
            h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply num_layers by 2 for bidirectional LSTM
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Multiply num_layers by 2 for bidirectional LSTM

            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
<<<<<<< HEAD
    
=======

>>>>>>> 3ec5449 (Added lSTM)
    # Model parameters
<<<<<<< HEAD
    input_size = 3
    hidden_size = 64
    num_layers = 2
<<<<<<< HEAD
    num_classes = len(data['Category'].unique())
=======
=======
    input_size = 4
    hidden_size = 128
    num_layers = 5
>>>>>>> a9f5826 (Updated LSTM Accuracy still low)
    num_classes = len(data["Category"].unique())
>>>>>>> 3ec5449 (Added lSTM)

    # Initialize model, loss function, and optimizer
    model = CrimeLSTM(input_size, hidden_size, num_layers, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Check if multiple GPUs are available and wrap the model with nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    # Move the model to the GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

<<<<<<< HEAD

=======
>>>>>>> 3ec5449 (Added lSTM)
    # Train the model
    num_epochs = 10

    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(dataloader):
            features = features.unsqueeze(1).to(device)
            labels = labels.to(device)
<<<<<<< HEAD
            
=======

>>>>>>> 3ec5449 (Added lSTM)
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

<<<<<<< HEAD
            if (i+1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(dataloader), loss.item()))
=======
            if (i + 1) % 10 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, len(dataloader), loss.item()
                    )
                )
>>>>>>> 3ec5449 (Added lSTM)

    # Test the model
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in test_dataloader:
            features = features.unsqueeze(1)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    # Calculate accuracy, confusion matrix, and classification report
    accuracy = accuracy_score(all_labels, all_predictions)
    conf_matrix = confusion_matrix(all_labels, all_predictions)
<<<<<<< HEAD
    class_report = classification_report(all_labels, all_predictions, target_names=encoder.classes_, zero_division=1)

=======
    class_report = classification_report(
        all_labels, all_predictions, target_names=encoder.classes_, zero_division=1
    )
>>>>>>> 3ec5449 (Added lSTM)

    print("Accuracy: {:.2f}".format(accuracy))
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # Create a function to inverse_transform the data
    def inverse_transform_data(data, scaler, encoder):
        inv_data = data.copy()
<<<<<<< HEAD
        inv_data[['Datetime', 'Latitude', 'Longitude']] = scaler.inverse_transform(data[['Datetime', 'Latitude', 'Longitude']])
        inv_data['Category'] = encoder.inverse_transform(data['Category'])
=======
        inv_data[["Datetime", "Latitude", "Longitude"]] = scaler.inverse_transform(
            data[["Datetime", "Latitude", "Longitude"]]
        )
        inv_data["Category"] = encoder.inverse_transform(data["Category"])
>>>>>>> 3ec5449 (Added lSTM)
        return inv_data

    # Inverse_transform test_data
    test_data_inv = inverse_transform_data(test_data, scaler, encoder)

    # Save the trained model
    model_path = "lstm_model.pt"
    torch.save(model.state_dict(), model_path)

<<<<<<< HEAD

=======
>>>>>>> 3ec5449 (Added lSTM)
    from folium.plugins import MarkerCluster

    # ...

    # Create a map centered at the mean latitude and longitude
<<<<<<< HEAD
    m = folium.Map(location=[test_data_inv['Latitude'].mean(), test_data_inv['Longitude'].mean()], zoom_start=12)
=======
    m = folium.Map(
        location=[test_data_inv["Latitude"].mean(), test_data_inv["Longitude"].mean()],
        zoom_start=12,
    )
>>>>>>> 3ec5449 (Added lSTM)

    # Create a MarkerCluster
    marker_cluster = MarkerCluster().add_to(m)

    counter = 0
    for idx, row in test_data_inv.iterrows():
<<<<<<< HEAD
        if row['Category'] == encoder.inverse_transform([all_predictions[counter]])[0]:
            marker_color = 'green'
        else:
            marker_color = 'red'
        folium.Marker([row['Latitude'], row['Longitude']], popup=row['Category'], icon=folium.Icon(color=marker_color)).add_to(marker_cluster)
        counter += 1


    # Save the map as an HTML file
    m.save("crime_map.html")
=======
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


>>>>>>> 3ec5449 (Added lSTM)
@stub.local_entrypoint()
def main():
    t0 = time.time()
    data = pd.read_csv("San_Francisco.csv")

    gpu_function.call(data)
<<<<<<< HEAD
    print("Time taken: {:.2f} seconds".format(time.time() - t0))
=======
    print("Time taken: {:.2f} seconds".format(time.time() - t0))
>>>>>>> 3ec5449 (Added lSTM)
