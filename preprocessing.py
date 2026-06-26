import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os

INPUT_X = 'data/dataX.pkl'
INPUT_Y = 'data/dataY.pkl'
TARGET_RES = (128, 128) 
TRAIN_RATIO = 0.85       

def load_pickle_data(path):
    with open(path, 'rb') as f:
        try:
            data = pickle.load(f, encoding='latin1')
        except:
            data = pickle.load(f)
            
    if isinstance(data, list):
        data = np.array(data)

    tensor = torch.tensor(data, dtype=torch.float32)
    return tensor

def main():
    print(f"--- Starting Preprocessing ---")

    print("Loading pickle files (this might take 10 seconds)...")
    x_raw = load_pickle_data(INPUT_X)
    y_raw = load_pickle_data(INPUT_Y)
    
    print(f"Original X Shape: {x_raw.shape}")
    print(f"Original Y Shape: {y_raw.shape}")

    print(f"\nResizing data to {TARGET_RES}...")
    x_resized = F.interpolate(x_raw, size=TARGET_RES, mode='bilinear', align_corners=False)
    y_resized = F.interpolate(y_raw, size=TARGET_RES, mode='bilinear', align_corners=False)
    
    print(f"New Shape: {x_resized.shape}")

    n_samples = x_resized.shape[0]
    n_train = int(n_samples * TRAIN_RATIO)
    
    train_x = x_resized[:n_train]
    train_y = y_resized[:n_train]
    
    test_x = x_resized[n_train:]
    test_y = y_resized[n_train:]
    
    print(f"\n--- Split Summary ---")
    print(f"Training Samples: {train_x.shape[0]}")
    print(f"Testing Samples:  {test_x.shape[0]}")

    os.makedirs('data/processed', exist_ok=True)
    
    torch.save(train_x, 'data/processed/train_x.pt')
    torch.save(train_y, 'data/processed/train_y.pt')
    torch.save(test_x,  'data/processed/test_x.pt')
    torch.save(test_y,  'data/processed/test_y.pt')
    
    print("\n Success! Clean data saved to 'data/processed/'")

if __name__ == "__main__":
    main()