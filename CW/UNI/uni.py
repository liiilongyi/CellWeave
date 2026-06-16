import os
import glob
import gc
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import json

Image.MAX_IMAGE_PIXELS = None  # Disable limit

#########################################
# 1. Configuration
#########################################

# --- Input Paths ---
# ==================== 1.1 Dynamically Read Global Configuration ====================
config_path = "../pipeline_config.json"
sample_id = "mouse_brain_demo"
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sample_id = cfg.get("SAMPLE_ID", "mouse_brain_demo")
    IMAGE_PATH = cfg.get("IMAGE_PATH", "mouse_brain_demo.png")
    POS_PATH = cfg.get("POS_PATH", "binned_outputs/square_002um/spatial/tissue_positions.parquet")
IMAGE_PATH = f'../{IMAGE_PATH}'
POS_PATH = f'../{POS_PATH}'
FINAL_MATRIX_PATH = f"final_matrix_{sample_id}.npy"
FINAL_BARCODES_PATH = f"final_barcodes_{sample_id}.npy"

LOCAL_MODEL_DIR = "./checkpoints/uni"
LOCAL_CKPT = os.path.join(LOCAL_MODEL_DIR, "pytorch_model.bin")
TEMP_DIR = "./uni_embeddings_chunks_temp"

# --- Model and Image Parameters ---
PIXEL_SCALE = 1
RAW_CROP_SIZE = 32
PATCH_SIZE = 224  # UNI model input size must be 224
EMBEDDING_DIM = 1024  # UNI ViT-Large output dimension
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Performance Parameters ---
BATCH_SIZE = 128  # Adjust based on VRAM (512-1024 works for 24GB VRAM)
NUM_WORKERS = 8  # Number of CPU worker processes
SAVE_EVERY = 10000  # Save a temporary file every N bins


#########################################
# 2. Dataset Definition
#########################################

class SpatialBinDataset(Dataset):
    def __init__(self, df, img_obj, transform, scale, patch_size):
        self.barcodes = df["barcode"].values
        # Precompute center coordinates
        self.centers_r = (df["pxl_row_in_fullres"].values * scale).astype(int)
        self.centers_c = (df["pxl_col_in_fullres"].values * scale).astype(int)

        self.img = img_obj
        self.transform = transform
        self.patch_size = patch_size
        self.half = patch_size // 2

    def __len__(self):
        return len(self.barcodes)

    def __getitem__(self, idx):
        # Get center coordinates
        r, c = self.centers_r[idx], self.centers_c[idx]
        barcode = self.barcodes[idx]

        # Calculate crop region (centered on the bin, cropping 224x224)
        left = c - self.half
        top = r - self.half
        right = left + self.patch_size
        bottom = top + self.patch_size

        # Crop the image
        patch = self.img.crop((left, top, right, bottom))

        # Handle rare edge cases with resize
        if patch.size != (self.patch_size, self.patch_size):
            patch = patch.resize((self.patch_size, self.patch_size), Image.BILINEAR)

        # Convert to Tensor
        if self.transform:
            patch_tensor = self.transform(patch)

        return patch_tensor, barcode


#########################################
# 3. Model Loader
#########################################

def load_uni_model():
    print(f"Loading UNI model from: {LOCAL_CKPT}")
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=224,
        patch_size=16,
        num_classes=0,  # num_classes=0 indicates feature extraction without classification
        dynamic_img_size=True,
    )

    # Load weights
    sd = torch.load(LOCAL_CKPT, map_location="cpu")
    msg = model.load_state_dict(sd, strict=False)
    print(f"Model loaded. Missing keys (expected for backbone only): {len(msg.missing_keys)}")

    model.eval().to(DEVICE)

    return model


#########################################
# 4. Main Pipeline
#########################################

def run_inference_and_save_chunks():
    """Phase 1: Run model inference and save results in chunks"""

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    print("PHASE 1: Inference & Chunk Saving")

    # --- 1. Read position information ---
    print("Reading position file...")
    df = pd.read_parquet(POS_PATH)
    df = df[df["in_tissue"] == 1].reset_index(drop=True)
    print(f"Total 'in_tissue' bins: {len(df)}")

    # --- 2. Load large image ---
    print("Loading Image into Memory...")
    full_image = Image.open(IMAGE_PATH).convert("RGB")
    full_image.load()

    # --- 3. Prepare DataLoader ---
    tf = transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = SpatialBinDataset(df, full_image, tf, PIXEL_SCALE, RAW_CROP_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --- 4. Load Model ---
    model = load_uni_model()

    # --- 5. Inference loop ---
    print("Starting Inference...")
    temp_dict = {}
    chunk_counter = 0
    total_processed = 0

    with torch.no_grad():
        for batch_imgs, batch_barcodes in tqdm(loader, desc="Inference"):
            batch_imgs = batch_imgs.to(DEVICE)

            # Mixed precision inference
            with torch.cuda.amp.autocast():
                feats = model(batch_imgs)  # [B, 1024]

            # Convert to Numpy
            feats_np = feats.float().cpu().numpy()

            # Store in dictionary
            for bc, emb in zip(batch_barcodes, feats_np):
                temp_dict[bc] = emb  # Store numpy array directly to save conversion time

            # Save chunks
            if len(temp_dict) >= SAVE_EVERY:
                save_name = os.path.join(TEMP_DIR, f"temp_chunk_{chunk_counter:05d}.npy")
                np.save(save_name, temp_dict)
                chunk_counter += 1
                total_processed += len(temp_dict)
                temp_dict = {}  # Reset
                gc.collect()

    # Save the remaining part
    if len(temp_dict) > 0:
        save_name = os.path.join(TEMP_DIR, f"temp_chunk_{chunk_counter:05d}.npy")
        np.save(save_name, temp_dict)
        chunk_counter += 1
        total_processed += len(temp_dict)

    print(f"Inference Done. Total chunks generated: {chunk_counter}")

    # Release VRAM
    del model
    del loader
    del full_image
    torch.cuda.empty_cache()
    gc.collect()


def merge_chunks_and_cleanup():
    """Phase 2: Merge chunks and clean up temporary files"""

    print("PHASE 2: Merging & Cleaning")

    chunk_files = sorted(glob.glob(os.path.join(TEMP_DIR, "temp_chunk_*.npy")))
    if not chunk_files:
        print("No chunk files found!")
        return

    # --- 1. Scan total sample count ---
    # Ensure this variable is initialized to 0 here
    total_samples = 0

    print("Scanning chunks to count samples...")
    for cf in tqdm(chunk_files, desc="Scanning"):
        # Only load objects to calculate the length
        chunk_data = np.load(cf, allow_pickle=True).item()
        total_samples += len(chunk_data)
        del chunk_data

    print(f"Total samples to merge: {total_samples}")

    # --- 2. Create Memmap matrix ---
    print(f"Creating memory-mapped file: {FINAL_MATRIX_PATH}")
    # The shape here is (N, 1024)
    fp = np.memmap(FINAL_MATRIX_PATH, dtype='float32', mode='w+', shape=(total_samples, EMBEDDING_DIM))

    all_barcodes = []
    current_idx = 0

    # --- 3. Merge and delete one by one ---
    for cf in tqdm(chunk_files, desc="Merging & Deleting"):
        # Load
        chunk_data = np.load(cf, allow_pickle=True).item()

        barcodes = list(chunk_data.keys())
        embeddings = np.array(list(chunk_data.values()), dtype='float32')

        n_items = len(barcodes)

        # Validate dimensions
        if embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Dimension mismatch! Expected {EMBEDDING_DIM}, got {embeddings.shape[1]}")

        # Write to disk matrix (memmap)
        fp[current_idx: current_idx + n_items, :] = embeddings

        # Collect Barcodes
        all_barcodes.extend(barcodes)

        current_idx += n_items

        # Clean up memory
        del chunk_data
        del embeddings

        # *** Key: Remove temporary files ***
        os.remove(cf)

    # Flush and close Memmap
    fp.flush()
    del fp

    # --- 4. Save Barcodes ---
    print(f"Saving barcodes to {FINAL_BARCODES_PATH} ...")
    np.save(FINAL_BARCODES_PATH, np.array(all_barcodes))

    # --- 5. Remove temporary directory ---
    if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
        os.rmdir(TEMP_DIR)
        print("Temporary directory removed.")
    else:
        print(f"Note: Temporary directory {TEMP_DIR} not empty or not removed.")

    print("ALL FINISHED!")
    print(f"Matrix saved: {FINAL_MATRIX_PATH} | Shape: ({total_samples}, {EMBEDDING_DIM})")
    print(f"Barcodes saved: {FINAL_BARCODES_PATH}")


if __name__ == "__main__":
    # Run inference
    run_inference_and_save_chunks()

    # Run merge
    merge_chunks_and_cleanup()