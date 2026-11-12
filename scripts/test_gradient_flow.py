import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from facial_emotion.model import create_model, preprocess_face_crops
from facial_emotion.config import EMOTION_CLASSES
from project_config import MELD_PREPARED_ROOT

train_manifest = pd.read_csv(MELD_PREPARED_ROOT / "reports" / "train" / "video_manifest.csv")
train_cache = Path("models/facial_emotion/face_cache/train")
class_to_idx = {name: i for i, name in enumerate(EMOTION_CLASSES)}

crops, labels = [], []
for _, row in train_manifest.head(200).iterrows():
    p = train_cache / (str(row["sample_id"]) + ".npy")
    if p.is_file():
        crops.append(np.load(p))
        labels.append(class_to_idx[row["emotion"].lower()])
    if len(crops) >= 100:
        break

crops = np.stack(crops)
labels = torch.tensor(labels)
tensor = preprocess_face_crops(crops, torch)

model = create_model()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print("Step-by-step loss and gradient norms:")
for step in range(5):
    optimizer.zero_grad()
    logits = model(tensor)
    loss = criterion(logits, labels)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    print(f"  Step {step}: Loss = {loss.item():.4f}, Grad Norm = {grad_norm.item():.4f}")

