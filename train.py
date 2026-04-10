import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from model import get_model
from utils import iou, dice
import torch.nn as nn
import torch.optim as optim
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

# Check dataset exists
if not os.path.exists(IMAGE_DIR) or not os.path.exists(MASK_DIR):
    raise Exception("Dataset not found! Create data/images and data/masks first.")

dataset = SegmentationDataset(IMAGE_DIR, MASK_DIR)

if len(dataset) == 0:
    raise Exception("Dataset is empty!")

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = get_model().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 3

print("Starting training...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# SAVE MODEL
print("Saving model...")
torch.save(model.state_dict(), "model.pth")
print("Model saved successfully!")

# Quick evaluation
model.eval()
with torch.no_grad():
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)

        print("IoU:", iou(preds, masks).item())
        print("Dice:", dice(preds, masks).item())
        break