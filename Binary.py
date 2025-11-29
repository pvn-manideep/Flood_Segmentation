import os
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import cv2

from models.vit_unet import Vit_UNet


# ======================================================
# 🔧 CONFIG
# ======================================================
MODEL_PATH = r"exp\model_vit_unet\vit_unet_weights_binary.pth"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ======================================================
# 🚀 FLASK SETUP
# ======================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


# ======================================================
# 🧠 LOAD MULTI-CLASS MODEL (Binary derived)
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("📥 Loading model checkpoint:", MODEL_PATH)
state_dict = torch.load(MODEL_PATH, map_location=device)

NUM_CLASSES = 10  # original training classes

model = Vit_UNet(img_ch=3, output_ch=NUM_CLASSES)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()

print("✅ Multi-class model loaded successfully (Binary mode)")


# ======================================================
# FLOOD CLASSES (Water + Pool)
# ======================================================
FLOOD_CLASSES = [5, 8]


# ======================================================
# 🖼️ PREPROCESS
# ======================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4093, 0.4471, 0.3405],
                         std=[0.1914, 0.1762, 0.1936])
])


# ======================================================
# 🎨 BINARY COLOR MAP (CYAN)
# ======================================================
BINARY_COLORS = np.array([
    [0, 0, 0],          # background = black
    [255, 255, 255]       # flood = white (visible clearly)
], dtype=np.uint8)


# ======================================================
# ROUTES
# ======================================================
@app.route("/")
def index():
    return render_template("index_binary.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Load image
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict multi-class mask
    with torch.no_grad():
        logits = model(img_tensor)

    preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Convert multi-class to binary mask
    binary_mask = np.isin(preds, FLOOD_CLASSES).astype(np.uint8)

    # Resize to original
    binary_resized = np.array(
        Image.fromarray(binary_mask).resize((w, h), Image.NEAREST)
    )

    # Colorize
    colored_mask = BINARY_COLORS[binary_resized]

    # Save mask
    mask_path = os.path.join(RESULT_FOLDER, "mask_" + file.filename)
    Image.fromarray(colored_mask).save(mask_path)

    # Save overlay
    overlay = cv2.addWeighted(np.array(image), 0.65, colored_mask, 0.35, 0)
    overlay_path = os.path.join(RESULT_FOLDER, "overlay_" + file.filename)
    Image.fromarray(overlay).save(overlay_path)

    print("🎯 Binary prediction completed:", file.filename)

    return render_template(
        "index_binary.html",
        image_path=img_path,
        result_path=mask_path,
        overlay_path=overlay_path
    )


# ======================================================
# RUN
# ======================================================
if __name__ == "__main__":
    app.run(debug=False, port=5001, use_reloader=False)
