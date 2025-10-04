import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json

# Hiperparámetros
WINDOW_SIZE = 24
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

# Detectar automáticamente el número de features
def get_n_features():
    try:
        X = np.load('processed_data/X_dsa.npy')
        n_features = X.shape[2]
        print(f"🔍 Detectados {n_features} features en el dataset")
        return n_features
    except:
        print("⚠️ No se pudo cargar X_dsa.npy, usando 12 features por defecto")
        return 12

N_FEATURES = get_n_features()

# Modelo DSA básico (LSTM secuencial)
class DSAModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # último estado oculto
        return self.fc(out)

if __name__ == "__main__":
    # Cargar datos
    X = np.load('processed_data/X_dsa.npy')
    y = np.load('processed_data/y_dsa.npy')

    # Normalización por-feature (mean/std) y guardar scaler
    feat_mean = X.reshape(-1, X.shape[2]).mean(axis=0)
    feat_std = X.reshape(-1, X.shape[2]).std(axis=0) + 1e-6
    X = (X - feat_mean) / feat_std

    # Guardar scaler y feature_cols para inferencia
    scaler = {
        'mean': feat_mean.tolist(),
        'std': feat_std.tolist(),
        'n_features': int(X.shape[2])
    }
    with open('processed_data/scaler.json', 'w') as f:
        json.dump(scaler, f)
    print("💾 Scaler guardado en processed_data/scaler.json")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    # Dataset y DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Instanciar modelo
    model = DSAModel(input_size=N_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Entrenamiento
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss/len(dataset):.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), 'processed_data/dsa_model.pt')
    print("Modelo DSA entrenado y guardado en processed_data/dsa_model.pt")
