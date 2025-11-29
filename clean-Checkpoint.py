import torch
import torch.serialization

# ---- TEMPORARY IMPORTS FOR CLEANING ONLY ----
from core.opt import Opt
torch.serialization.add_safe_globals([Opt])
# ----------------------------------------------

INPUT_CKPT = r"C:\Users\pvnma\Desktop\pp1\b,c\exp\model_vit_unet\vit_unet_best.pth"
OUTPUT_CKPT = r"C:\Users\pvnma\Desktop\pp1\b,c\exp\model_vit_unet\vit_unet_weights_only.pth"

print("📥 Loading original checkpoint (with safe_globals enabled)...")

ckpt = torch.load(INPUT_CKPT, map_location="cpu", weights_only=False)

print("Checkpoint keys:", ckpt.keys())

# extract only weights
weights = ckpt["model"]

print("💾 Saving weights-only checkpoint...")
torch.save(weights, OUTPUT_CKPT)

print("✅ CLEAN WEIGHTS SAVED SUCCESSFULLY!")
print("➡️ New file:", OUTPUT_CKPT)
