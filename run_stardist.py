import os
import time
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile  # <--- Key change: Direct import of tifffile
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from stardist import random_label_cmap
import tensorflow as tf
from PIL import Image
from skimage.io import imread, imsave
import json

Image.MAX_IMAGE_PIXELS = None

# ================= Configuration Area =================
# Input image path
# ==================== Dynamically Read Global Configuration ====================
config_path = "pipeline_config.json"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    INPUT_IMAGE_PATH = cfg.get("IMAGE_PATH", "mouse_brain_demo.png")
    sample_id = cfg.get("SAMPLE_ID", "mouse_brain_demo")
    filename = os.path.basename(INPUT_IMAGE_PATH).split('.')[0]
    OUTPUT_DIR = "./stardist_npy"

MIN_CELL_SIZE = 10
MAX_CELL_SIZE = 5000
# ===================================================

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def setup_env():
    """Set up execution environment"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Detected GPUs: {len(gpus)}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU detected, falling back to CPU (processing will be slow, thank you for your patience)")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def read_ome_tiff(path):
    """
    Dedicated function for reading Xenium OME-TIFF files
    Prioritizes reading the largest Series (the highest resolution level)
    """
    print(f"   Parsing OME-TIFF using tifffile: {path}")
    try:
        with tifffile.TiffFile(path) as tif:
            # Series 0 is typically the highest resolution for Xenium OME-TIFF files
            # To be safe, we iterate through all series to find the one with the maximum pixel count
            largest_series = max(tif.series, key=lambda s: s.size)

            print(f"   Detected {len(tif.series)} levels, selected shape: {largest_series.shape}")

            # Read data
            img = largest_series.asarray()
            return img
    except Exception as e:
        print(f"Read failed: {e}")
        raise e


def process_and_filter_labels(raw_labels, min_size, max_size):
    from skimage.measure import label  # Deferred import
    print(f"\n[Post-processing] Resetting connected components and filtering outliers...")

    # 1. Force ID reset
    refined_labels = label(raw_labels > 0, connectivity=1).astype(np.uint32)

    # 2. Filter sizes
    ids, sizes = np.unique(refined_labels, return_counts=True)
    background_mask = (ids == 0)
    # Remove any objects that fall outside the min~max range
    to_remove = ids[((sizes < min_size) | (sizes > max_size)) & (~background_mask)]

    if len(to_remove) > 0:
        print(f"   Removed {len(to_remove)} objects with anomalous sizes")
        mask_remove = np.isin(refined_labels, to_remove)
        refined_labels[mask_remove] = 0

    return refined_labels


def save_preview(img, labels, output_path):
    try:
        # If multi-channel (H, W, C), take the mean to convert to grayscale for display, or take only the first channel
        if img.ndim == 3:
            img_show = img.mean(axis=2).astype(np.uint8)
        else:
            img_show = img

        h, w = labels.shape
        cy, cx = h // 2, w // 2
        crop_size = 500
        # Prevent out-of-bounds
        y1, y2 = max(0, cy - crop_size), min(h, cy + crop_size)
        x1, x2 = max(0, cx - crop_size), min(w, cx + crop_size)

        img_crop = img_show[y1:y2, x1:x2]
        lbl_crop = labels[y1:y2, x1:x2]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(img_crop, cmap='gray')
        plt.title("Raw Image (Center Crop)")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_crop, cmap='gray')
        if lbl_crop.max() > 0:
            lbl_masked = np.ma.masked_where(lbl_crop == 0, lbl_crop)
            plt.imshow(lbl_masked, cmap=random_label_cmap(), alpha=0.5)
        plt.title("Segmentation")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"   Preview image saved")
    except Exception as e:
        print(f"    Failed to generate preview image: {e}")


def main():
    start_total = time.time()
    setup_env()

    # 1. Read image
    print("\n[1/5] Reading image...")
    if not os.path.exists(INPUT_IMAGE_PATH):
        print(f"Error: File not found {INPUT_IMAGE_PATH}")
        return

    # Read using tifffile
    # img = read_ome_tiff(INPUT_IMAGE_PATH)
    img = imread(INPUT_IMAGE_PATH)
    print(f"   Image loaded successfully, shape: {img.shape}, dtype: {img.dtype}")

    # 2. Normalization
    print("\n[2/5] Normalizing (this may require a significant amount of memory)...")
    # Note: StarDist normalize can be memory-intensive. For extremely large images, tiling or float16 is recommended. Standard routine is used here.
    img_norm = normalize(img, 1, 99.8, axis=(0, 1))

    # 3. Load Model & Prediction
    print("\n[3/5] Loading model and predicting...")
    try:
        model = StarDist2D.from_pretrained('2D_versatile_he')
    except:
        print("   Unable to automatically download model, please check connection or specify the path manually")
        return

    # Configuration parameters (only setting up parameters, prediction logic is handled separately below)
    if img_norm.ndim == 3:
        print("   Mode: RGB (YXC)")
        axes_param = 'YXC'
        n_channels = img_norm.shape[2]
        block_size_param = (2048, 2048, n_channels)
        min_overlap_param = (128, 128, 0)
        context_param = (128, 128, 0)
    else:
        print("   Mode: Grayscale (YX)")
        axes_param = 'YX'
        block_size_param = (2048, 2048)
        min_overlap_param = (128, 128)
        context_param = (128, 128)

    # ==================== Key Change: Dynamically select prediction method ====================
    # Get width and height of the image (first two dimensions)
    img_h, img_w = img_norm.shape[0], img_norm.shape[1]

    # If image size is less than or equal to tile size (2048), predict directly using predict_instances (no tiling)
    if img_h <= 2048 and img_w <= 2048:
        print(f"   Image size is {img_h}x{img_w}. Suitable for standard prediction (no tiling needed)...")
        labels_raw, _ = model.predict_instances(
            img_norm,
            axes=axes_param,
            prob_thresh=0.2,
            nms_thresh=0.1,
        )
    else:
        # If image is larger than 2048, enable tiled prediction for large images
        print(f"   Image size is {img_h}x{img_w}. Detected very large image, starting tiled prediction...")
        labels_raw, _ = model.predict_instances_big(
            img_norm,
            axes=axes_param,
            block_size=block_size_param,
            min_overlap=min_overlap_param,
            context=context_param,
            prob_thresh=0.2,
            nms_thresh=0.1,
            show_progress=True  # Kept as tiling supports this parameter
        )
    # ==========================================================================================
    # 4. Post-processing
    print("\n[4/5] Performing post-processing fixes...")
    labels_final = process_and_filter_labels(
        labels_raw,
        min_size=MIN_CELL_SIZE,
        max_size=MAX_CELL_SIZE
    )

    # 5. Save results
    print("\n[5/5] Saving results...")

    save_path_npy = os.path.join(OUTPUT_DIR, f"{sample_id}_labels.npy")
    np.save(save_path_npy, labels_final)
    print(f"   NPY Mask: {save_path_npy}")

    # Preview
    save_path_png = os.path.join(OUTPUT_DIR, f"{sample_id}_preview.png")
    save_preview(img, labels_final, save_path_png)

    print("\n" + "=" * 30)
    print(f"Total time elapsed: {(time.time() - start_total) / 60:.1f} minutes")
    print("=" * 30)

    # Clean up memory
    del model, img, img_norm, labels_raw, labels_final
    gc.collect()


if __name__ == "__main__":
    main()