import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, redirect, url_for

from models.vit_unet import Vit_UNet


# ======================================================
# PATHS (RELATIVE → GOOD FOR DEPLOYMENT)
# ======================================================
MODEL_PATH_BINARY = r"exp/model_vit_unet/vit_unet_weights_Binary.pth"
MODEL_PATH_COLOR  = r"exp/model_vit_unet/vit_unet_weights_Color.pth"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# ======================================================
# FLASK APP
# ======================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


# ======================================================
# DEVICE
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥 Device:", device)


# ======================================================
# PREPROCESS (SHARED)
# ======================================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4093, 0.4471, 0.3405],
                         std=[0.1914, 0.1762, 0.1936]),
])


# ======================================================
# MODELS
# ======================================================
print("📥 Loading BINARY checkpoint:", MODEL_PATH_BINARY)
state_binary = torch.load(MODEL_PATH_BINARY, map_location=device)
model_binary = Vit_UNet(img_ch=3, output_ch=10)   # 10-class head
model_binary.load_state_dict(state_binary, strict=False)
model_binary.to(device).eval()
print("✅ Binary model loaded (using multi-class head for flood extraction).")

print("📥 Loading COLOR checkpoint:", MODEL_PATH_COLOR)
state_color = torch.load(MODEL_PATH_COLOR, map_location=device)
model_color = Vit_UNet(img_ch=3, output_ch=10)
model_color.load_state_dict(state_color, strict=False)
model_color.to(device).eval()
print("✅ Colour model loaded.")


# ======================================================
# BINARY LOGIC (Flood from multi-class)
# ======================================================
FLOOD_CLASSES = [5, 8]  # water + pool

BINARY_COLORS = np.array([
    [0, 0, 0],          # background
    [255, 255, 255],    # flood → white
], dtype=np.uint8)


def run_binary_segmentation(pil_img, filename):
    w, h = pil_img.size
    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model_binary(x)                 # [1,10,H,W]
        pred = torch.argmax(out, dim=1)[0]    # [H,W]
        pred = pred.cpu().numpy().astype(np.uint8)

    # flood = classes 5 or 8
    binary_mask = np.isin(pred, FLOOD_CLASSES).astype(np.uint8)

    # resize to original
    binary_resized = np.array(
        Image.fromarray(binary_mask).resize((w, h), Image.NEAREST)
    )

    colored_mask = BINARY_COLORS[binary_resized]

    mask_path = os.path.join(RESULT_FOLDER, "mask_bin_" + filename)
    Image.fromarray(colored_mask).save(mask_path)

    overlay = cv2.addWeighted(
        np.array(pil_img), 0.65, colored_mask, 0.35, 0
    )
    overlay_path = os.path.join(RESULT_FOLDER, "overlay_bin_" + filename)
    Image.fromarray(overlay).save(overlay_path)

    return mask_path, overlay_path


# ======================================================
# COLOR / MULTI-CLASS LOGIC
# ======================================================

# 0 background, 1 water, 2 vegetation, 3 built-up
COLOR_MAP = np.array([
    [0, 0, 0],          # 0 Background
    [0, 120, 255],      # 1 Water / Pool (blue)
    [0, 255, 60],       # 2 Vegetation (green)
    [255, 0, 170],      # 3 Built-up (pink)
], dtype=np.uint8)

WATER   = [5, 8]
VEG     = [6, 9]
BUILTUP = [1, 2, 3, 4, 7]


def run_color_segmentation(pil_img, filename):
    w, h = pil_img.size
    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model_color(x)
        pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)

    pred_resized = np.array(
        Image.fromarray(pred).resize((w, h), Image.NEAREST)
    )

    # remap 0..9 → 0..3
    remap = np.zeros_like(pred_resized, dtype=np.uint8)

    for c in WATER:
        remap[pred_resized == c] = 1
    for c in VEG:
        remap[pred_resized == c] = 2
    for c in BUILTUP:
        remap[pred_resized == c] = 3

    mask = COLOR_MAP[remap]

    mask_path = os.path.join(RESULT_FOLDER, "mask_col_" + filename)
    Image.fromarray(mask).save(mask_path)

    overlay = cv2.addWeighted(
        np.array(pil_img), 0.6, mask, 0.4, 0
    )
    overlay_path = os.path.join(RESULT_FOLDER, "overlay_col_" + filename)
    Image.fromarray(overlay).save(overlay_path)

    return mask_path, overlay_path


# ======================================================
# ROUTES
# ======================================================

@app.route("/")
def home():
    # simple main menu
    return render_template("main_menu.html")


@app.route("/binary", methods=["GET", "POST"])
def binary_page():
    if request.method == "GET":
        return render_template("index_binary.html")

    # POST → run prediction
    if "file" not in request.files:
        return redirect(url_for("binary_page"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("binary_page"))

    filename = file.filename
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    image = Image.open(img_path).convert("RGB")

    mask_path, overlay_path = run_binary_segmentation(image, filename)

    return render_template(
        "index_binary.html",
        image_path=img_path,
        result_path=mask_path,
        overlay_path=overlay_path,
    )


@app.route("/colour", methods=["GET", "POST"])
def colour_page():
    if request.method == "GET":
        return render_template("index.html")

    if "file" not in request.files:
        return redirect(url_for("colour_page"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("colour_page"))

    filename = file.filename
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    image = Image.open(img_path).convert("RGB")

    mask_path, overlay_path = run_color_segmentation(image, filename)

    return render_template(
        "index.html",
        image_path=img_path,
        result_path=mask_path,
        overlay_path=overlay_path,
    )


# ======================================================
# ENTRYPOINT (for local dev; ignored by gunicorn)
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
