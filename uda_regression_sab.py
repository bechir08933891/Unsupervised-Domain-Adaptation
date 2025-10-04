# uda_regression_sab.py
# -*- coding: utf-8 -*-
"""
uda_regression_sab.py
Contains:
- DANNRegressor model
- Gradient Reversal Layer
- Training loop (runs only when executed directly)
- Prediction function (callable from app)
"""

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------- Paths ---------------------
PREPROCESSOR_PATH = "model/preprocessor.pkl"
MODEL_PATH = "model/dann_regressor.pth"

# --------------------- Model & GRL ---------------------
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def forward(self, x, alpha=1.0):
        return GradientReversalFunction.apply(x, alpha)

class DANNRegressor(nn.Module):
    def __init__(self, input_dim):
        super(DANNRegressor, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # regression head - ensure non-negative output with ReLU
        self.regressor = nn.Linear(64, 1)
        self.grl = GradientReversalLayer()
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature(x)
        regression_output = torch.relu(self.regressor(features))
        domain_output = self.domain_classifier(self.grl(features, alpha))
        return regression_output, domain_output

# --------------------- Prediction function (used by app) ---------------------
def predict_rows(df, model, preprocessor):
    """
    Predict heart_rate_apache for one or multiple rows (DataFrame).
    model : a loaded DANNRegressor instance
    preprocessor : fitted ColumnTransformer (from training)
    """
    # Ensure input columns exist in df (preprocessor.transform will fail otherwise)
    df_processed = preprocessor.transform(df)
    df_tensor = torch.tensor(df_processed, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds, _ = model(df_tensor, alpha=1.0)
    return preds.numpy().flatten()

# --------------------- TRAINING (only when script run directly) ---------------------
def train_and_save(epochs=50, lr=1e-3, batch_size=16):
    # Load datasets
    source_df = pd.read_csv('data/mcr_source.csv')
    target_df = pd.read_csv('data/mcr_target.csv')
    TARGET = 'heart_rate_apache'

    # Preprocessing columns
    categorical_cols = ['gender', 'ethnicity']
    numeric_cols = [col for col in source_df.columns if col not in categorical_cols + [TARGET]]

    # Fit preprocessor on source
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    X_source = preprocessor.fit_transform(source_df.drop(columns=[TARGET]))
    y_source = source_df[TARGET].values.reshape(-1, 1)
    X_target = preprocessor.transform(target_df)

    # Save preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved preprocessor to {PREPROCESSOR_PATH}")

    # Convert to tensors & loaders
    X_source_tensor = torch.tensor(X_source, dtype=torch.float32)
    y_source_tensor = torch.tensor(y_source, dtype=torch.float32)
    X_target_tensor = torch.tensor(X_target, dtype=torch.float32)

    source_loader = DataLoader(TensorDataset(X_source_tensor, y_source_tensor), batch_size=batch_size, shuffle=True)
    target_loader = DataLoader(TensorDataset(X_target_tensor), batch_size=batch_size, shuffle=True)

    # Instantiate model with correct input dim
    input_dim = X_source.shape[1]
    model = DANNRegressor(input_dim)

    # Losses & optimizer
    regression_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_reg_loss = 0.0
        total_domain_loss = 0.0

        p = epoch / max(1, epochs)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        for (X_s, y_s), (X_t,) in zip(source_loader, target_loader):
            optimizer.zero_grad()
            # source forward
            reg_output, domain_output_s = model(X_s, alpha=alpha)
            reg_loss = regression_criterion(reg_output, y_s)
            domain_labels_s = torch.zeros(X_s.size(0), dtype=torch.long)
            domain_loss_s = domain_criterion(domain_output_s, domain_labels_s)

            # target forward (domain only)
            _, domain_output_t = model(X_t, alpha=alpha)
            domain_labels_t = torch.ones(X_t.size(0), dtype=torch.long)
            domain_loss_t = domain_criterion(domain_output_t, domain_labels_t)

            loss = reg_loss + domain_loss_s + domain_loss_t
            loss.backward()
            optimizer.step()

            total_reg_loss += reg_loss.item()
            total_domain_loss += (domain_loss_s + domain_loss_t).item()

        if (epoch + 1) % 10 == 0 or epoch == epochs-1:
            print(f"Epoch [{epoch+1}/{epochs}] | Reg Loss: {total_reg_loss/len(source_loader):.4f} | Domain Loss: {total_domain_loss/len(source_loader):.4f} | Alpha: {alpha:.4f}")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

# --------------------- Helper to load preprocessor (for app) ---------------------
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)

# --------------------- When executed directly, train ---------------------
if __name__ == "__main__":
    train_and_save(epochs=50, lr=1e-3, batch_size=16)
