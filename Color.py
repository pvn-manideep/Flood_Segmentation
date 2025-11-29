import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for
from huggingface_hub import hf_hub_download
from models.vit_unet import Vit_UNet


# ======================================================
# CONFIG
# ======================================================
MODEL_PATH = hf_hub_download(
    repo_id="RemainCalm/flood-segmentation-vit-unet",
    filename="vit_unet_weights_Binary.pth",
)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ======================================================
# FLASK SETUP
# ======================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


# ======================================================
# LOAD MODEL
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("📥 Loading checkpoint:", MODEL_PATH)
state_dict = torch.load(MODEL_PATH, map_location=device)

NUM_CLASSES = 10
model = Vit_UNet(img_ch=3, output_ch=NUM_CLASSES)
model.load_state_dict(state_dict, strict=False)
model.to(device).eval()

print("✅ Multi-class model loaded successfully!")


# ======================================================
# PREPROCESS
# ======================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4093, 0.4471, 0.3405],
                         std=[0.1914, 0.1762, 0.1936])
])


# ======================================================
# PROFESSIONAL COLOR MAP
# ======================================================
# 0 background (black)
# 1 water/pool (blue)
# 2 vegetation (green)
# 3 buildings/roads/vehicles (pink)

COLOR_MAP = np.array([
    [0, 0, 0],          # 0 Background
    [0, 120, 255],      # 1 Water / Pool
    [0, 255, 60],       # 2 Vegetation
    [255, 0, 170]       # 3 Built-up Area
], dtype=np.uint8)

# Groups for 10-class → 4-color reduction
WATER = [5, 8]
VEG = [6, 9]
BUILTUP = [1, 2, 3, 4, 7]


# ======================================================
# LEGEND DRAWER
# ======================================================
def add_legend(image):
    legend = image.copy()
    box_w, box_h = 220, 140
    cv2.rectangle(legend, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)

    items = [
        ("Background", (0, 0, 0)),
        ("Water", (0, 120, 255)),
        ("Vegetation", (0, 255, 60)),
        ("Buildings/Roads", (255, 0, 170))
    ]

    y = 35
    for name, color in items:
        cv2.rectangle(legend, (20, y - 15), (50, y + 15), color, -1)
        cv2.putText(legend, name, (60, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 35

    return legend


# ======================================================
# ROUTES
# ======================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    img_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(img_path)

    # Load + preprocess
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    x = transform(image).unsqueeze(0).to(device)

    # MODEL PREDICTION
    with torch.no_grad():
        out = model(x)

    pred = torch.argmax(out, dim=1).cpu().numpy()[0]
    pred = np.array(Image.fromarray(pred.astype(np.uint8)).resize((w, h), Image.NEAREST))

    print("Unique classes:", np.unique(pred))

    # ======================================================
    # REMAP TO SUPER-CLASSES (0–3)
    # ======================================================
    remap = np.zeros_like(pred)

    for c in WATER:
        remap[pred == c] = 1
    for c in VEG:
        remap[pred == c] = 2
    for c in BUILTUP:
        remap[pred == c] = 3

    mask = COLOR_MAP[remap]

    # SAVE MASK
    mask_path = os.path.join(RESULT_FOLDER, "mask_" + file.filename)
    Image.fromarray(mask).save(mask_path)

    # OVERLAY + LEGEND
    overlay = cv2.addWeighted(np.array(image), 0.6, mask, 0.4, 0)

    overlay_path = os.path.join(RESULT_FOLDER, "overlay_" + file.filename)
    Image.fromarray(overlay).save(overlay_path)

    print("Multi-class prediction completed:", file.filename)

    return render_template(
        "index.html",
        image_path=img_path,
        result_path=mask_path,
        overlay_path=overlay_path
    )


# ======================================================
# RUN APP
# ======================================================
if __name__ == "__main__":
    app.run(debug=False, port=5002, use_reloader=False)
