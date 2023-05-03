import modal 
import time
import pandas as pd
stub = modal.Stub(
    "example-import-torch",
    image=modal.Image.debian_slim().pip_install(
        "pandas",
        "scikit-learn",
        "numpy",
        "imblearn",
        "torch", find_links="https://download.pytorch.org/whl/cu116",
    ),
)

@stub.function( timeout=60 * 60 * 6)
def processing(df):
  
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
    from imblearn.over_sampling import SMOTE
    from sklearn.metrics import accuracy_score
    from pandas.tseries.holiday import USFederalHolidayCalendar


    # Preprocess data (you'll need to adjust this to match your actual data)
    df['Date_Time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Year'] = df['Date_Time'].dt.year
    df['Month'] = df['Date_Time'].dt.month
    df['Day'] = df['Date_Time'].dt.day
    df['Time_minutes'] = df['Date_Time'].dt.hour * 60 + df['Date_Time'].dt.minute

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=df['Date_Time'].min().date(), end=df['Date_Time'].max().date())
    df['IsHoliday'] = df['Date_Time'].dt.date.isin(holidays)


    print(df['IsHoliday'].value_counts())
    return

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(df[['Day_of_Week', 'Part_of_Day']])
    one_hot_columns = one_hot_encoder.get_feature_names_out(['Day_of_Week', 'Part_of_Day'])

    scaler = StandardScaler()
    scaled_columns = ['Year', 'Month', 'Day', 'Time_minutes', 'IsHoliday', 'Latitude', 'Longitude']
    scaled_values = scaler.fit_transform(df[scaled_columns])

    X = np.hstack((scaled_values, one_hot_encoded))
    y = df['Category'].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Convert to tensors
    X_train_resampled = torch.tensor(X_train_resampled, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train_resampled = torch.tensor(y_train_resampled, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    class CrimeCategoryGRU(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
            super(CrimeCategoryGRU, self).__init__()
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            rnn_out, _ = self.rnn(x)
            last_out = rnn_out[:, -1, :]
            x = self.fc(last_out)
            return x

    input_dim = X_train.shape[1]
    seq_length = 1  # This will be changed if you have a different sequence length
    output_dim = len(np.unique(y_train))

    X_train_resampled = X_train_resampled.view(-1, seq_length, input_dim)
    X_test = X_test.view(-1, seq_length, input_dim)

    def train_and_evaluate(lr, hidden_dim, X_train, y_train, X_test, y_test):
        model = CrimeCategoryGRU(input_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        n_epochs = 1000

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')

        model.eval()
        with torch.no_grad():
            output = model(X_test)
            _, predicted = torch.max(output.data, 1)
            accuracy = accuracy_score(y_test, predicted)

        print(f'Accuracy: {accuracy}')
        return accuracy

    learning_rates = [0.0001]
    hidden_dims = [128, 256]

    best_accuracy = 0
    best_params = None

    for lr in learning_rates:
        for hidden_dim in hidden_dims:
            current_accuracy = train_and_evaluate(lr, hidden_dim, X_train_resampled, y_train_resampled, X_test, y_test)
            print(f'Learning rate: {lr}, Hidden dim: {hidden_dim}, Accuracy: {current_accuracy}')
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_params = (lr, hidden_dim)

    print(f'Best learning rate: {best_params[0]}, Best hidden_dim: {best_params[1]}')

@stub.local_entrypoint()
def main():
    t0 = time.time()
    data = pd.read_csv("San_Francisco.csv")

    processing.call(data)

    print(f"Total time: {time.time() - t0} seconds")