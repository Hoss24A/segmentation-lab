from flask import Flask, request, jsonify
import torch
from model import get_model
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get model path (fallback if .env missing)
MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = get_model()

if MODEL_PATH and os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("Warning: model file not found. Skipping load (CI mode).")

model.to(device)
model.eval()

@app.route("/")
def home():
    return "Segmentation API running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    image = Image.open(file).resize((256, 256))
    image = np.array(image) / 255.0

    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred).cpu().numpy()[0][0]

    return jsonify({
        "message": "Segmentation completed",
        "mask_shape": pred.shape
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)