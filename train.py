# Code for training the CFD Copilot model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os

from models.hard_constriant_and_Attention import PhysicsAttentionFNO
from models.hard_constriant_and_Attention import apply_hard_physics_constraint
from models.scaled_rmodel import UnitGaussianNormalizer

TRAIN_PATH_X = 'data/Processed/train_x.pt'
TRAIN_PATH_Y = 'data/Processed/train_y.pt'

MODES = 24        
WIDTH = 128         
BATCH_SIZE = 20
EPOCHS = 500
LEARNING_RATE = 0.001

print("--- Loading Data ---")
x_train = torch.load(TRAIN_PATH_X)
y_train = torch.load(TRAIN_PATH_Y)

x_normalizer = UnitGaussianNormalizer(x_train)
y_normalizer = UnitGaussianNormalizer(y_train)

# --- SAVE THE SCALERS ---
print("Saving scaler stats...")
scaler_state = {
    'x_mean': x_normalizer.mean,
    'x_std':  x_normalizer.std,
    'y_mean': y_normalizer.mean,
    'y_std':  y_normalizer.std
}
torch.save(scaler_state, 'checkpoints/attention_1.pth')
print("Scaler saved to checkpoints/attention_1.pth")

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Training on: {device} ---")

x_normalizer.to(device)
y_normalizer.to(device)

model = PhysicsAttentionFNO(max_modes=MODES, width=WIDTH, num_in_channels=x_train.shape[1], num_out_channels=y_train.shape[1]).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.5)

criterion = nn.MSELoss()

print("--- Starting Training Loop ---")
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x_raw, y_raw = x.to(device), y.to(device)

        x = x_normalizer.encode(x_raw)
        y = y_normalizer.encode(y_raw)

        optimizer.zero_grad()

        out = model(x)

        # Apply Hard Physics Constraint (No-Slip)
        out = apply_hard_physics_constraint(out, x_raw, y_normalizer)

        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    if (epoch+1) % 5 == 0:
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} \t Loss: {avg_loss:.6f} \t LR: {scheduler.get_last_lr()[0]:.6f}")

total_time = time.time() - start_time
print(f"--- Training Complete in {total_time:.2f} seconds ---")

os.makedirs("checkpoints", exist_ok=True)
save_path = "checkpoints/Attention_1.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")