import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from pathlib import Path
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import json
import os
import platform
import time

# =============================
# 1. Load trained model
# =============================
from models.unet_blindspot import UNetBlindspot as UNet

CKPT_PATH = Path("checkpoints/n2v_unet/ckpt_1600.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print("🔹 Loading model...")
    model = UNet(in_ch=3, base=48)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print("✅ Model loaded successfully.")
    return model

# =============================
# 2. Select an image
# =============================
def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff")]
    )
    root.destroy()
    if not file_path:
        raise FileNotFoundError("❌ No image selected.")
    print(f"📁 Selected image: {file_path}")
    return Path(file_path)

# =============================
# 3. Preprocess image
# =============================
def preprocess_image(img_path):
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = transforms.ToTensor()
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # pad image to multiple of 16 for UNet
    _, _, h, w = tensor.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    return tensor, np.array(img)

# =============================
# 4. Run model
# =============================
def denoise_image(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        output = torch.clamp(output, 0, 1)
    denoised = output.squeeze().permute(1, 2, 0).cpu().numpy()
    return denoised

# =============================
# 5. Compute analysis metrics
# =============================
def analyze_image(image_array):
    gray = np.mean(image_array, axis=2) if image_array.ndim == 3 else image_array
    mean, median, std = sigma_clipped_stats(gray, sigma=3.0)
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.0 * std)
    sources = daofind(gray - median)
    count = 0 if sources is None else len(sources)

    return {
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "num_stars": int(count)
    }

def compute_comparison_metrics(orig, denoised):
    gray_orig = np.mean(orig, axis=2) if orig.ndim == 3 else orig
    gray_denoised = np.mean(denoised, axis=2) if denoised.ndim == 3 else denoised

    # Clip to same shape
    min_h = min(gray_orig.shape[0], gray_denoised.shape[0])
    min_w = min(gray_orig.shape[1], gray_denoised.shape[1])
    gray_orig = gray_orig[:min_h, :min_w]
    gray_denoised = gray_denoised[:min_h, :min_w]

    diff = np.abs(gray_orig - gray_denoised)
    mae = np.mean(diff)
    mse = np.mean((gray_orig - gray_denoised) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")

    snr_orig = np.mean(gray_orig) / (np.std(gray_orig) + 1e-8)
    snr_denoised = np.mean(gray_denoised) / (np.std(gray_denoised) + 1e-8)
    brightness_change = ((np.mean(gray_denoised) - np.mean(gray_orig)) / np.mean(gray_orig)) * 100

    return {
        "mae": float(mae),
        "mse": float(mse),
        "psnr": float(psnr),
        "snr_original": float(snr_orig),
        "snr_denoised": float(snr_denoised),
        "brightness_change_percent": float(brightness_change)
    }

# =============================
# 6. Main function
# =============================
def main():
    os.makedirs("reports/predictions", exist_ok=True)
    start_time = time.time()
    model = load_model()
    img_path = select_image()

    tensor, orig = preprocess_image(img_path)
    denoised = denoise_image(model, tensor)

    stats_orig = analyze_image(orig)
    stats_denoised = analyze_image(denoised)
    comparison = compute_comparison_metrics(orig / 255.0, denoised)

    duration = time.time() - start_time

    # Save clean image
    clean_img = Image.fromarray((denoised * 255).astype(np.uint8))
    clean_path = Path("reports/predictions") / f"cleaned_{img_path.stem}.png"
    clean_img.save(clean_path)

    # Build JSON report
    json_data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": platform.platform(),
            "device": DEVICE,
            "processing_time_sec": round(duration, 3),
            "image_shape": list(orig.shape)
        },
        "original_image": str(img_path),
        "denoised_image": str(clean_path),
        "original_stats": stats_orig,
        "denoised_stats": stats_denoised,
        "comparison_metrics": comparison
    }

    json_path = Path("reports/predictions") / f"report_{img_path.stem}.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    # Display side-by-side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(orig)
    ax[0].set_title(f"Original ({stats_orig['num_stars']} stars)")
    ax[1].imshow(denoised)
    ax[1].set_title(f"Denoised ({stats_denoised['num_stars']} stars)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"✅ Denoised image saved to: {clean_path}")
    print(f"📊 Full JSON report saved to: {json_path}")

if __name__ == "__main__":
    main()
