import os
import re
import time
import random
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Compose, Resize, ScaleIntensity, RandFlip, RandAffine
)
from monai.networks.nets import UNet
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Paths
IMAGE_DIR = "/data_vault/hexai/OAICartilage/Images_Folders_jpg2_cropped"
MASK_DIR  = "/data_vault/hexai/OAICartilage/Annotations2"

# Settings
NUM_CLASSES = 5
TARGET_SHAPE = (112, 224, 224)  # (D, H, W)
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 2
LR = 1e-4
BEST_MODEL_PATH = "/home/feg48/2.5D_seg/final_results/3d_best_model.pth"

# Reproducibility
SEED = 65
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Dataset 
class KneeVolumeDataset3D(Dataset):
    def __init__(self, image_root_dir, mask_root_dir, study_ids, image_transform=None, mask_transform=None):
        self.image_root_dir = image_root_dir
        self.mask_root_dir = mask_root_dir
        self.study_ids = list(study_ids)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.study_ids)

    def _slice_key(self, fname: str):
        m = re.search(r'(?:_slice_)?(\d+)\.jpg$', fname)
        return int(m.group(1)) if m else -1

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        img_dir = os.path.join(self.image_root_dir, study_id)
        mask_path = os.path.join(self.mask_root_dir, f"{study_id}.nii.gz")
    

        # load mask volume (D, H, W)
        mask_vol = nib.load(mask_path).get_fdata().astype(np.int64)
        D_mask = mask_vol.shape[0]

        # collect and sort slice files
        files = sorted([f for f in os.listdir(img_dir) if f.endswith(".jpg")], key=self._slice_key)
        D_img = len(files)
        if D_img == 0:
            raise RuntimeError(f"No .jpg slices in {img_dir}")

        # align depths
        D = min(D_mask, D_img)

        imgs = []
        for s in range(D):
            img = Image.open(os.path.join(img_dir, files[s])).convert("L")
            img_np = np.asarray(img, dtype=np.float32) / 255.0  # (H, W)
            imgs.append(torch.from_numpy(img_np))

        masks = []
        for s in range(D):
            m_np = mask_vol[s, :, :]
            masks.append(torch.from_numpy(m_np.astype(np.int64)))

        # stack to volumes
        img_vol = torch.stack(imgs, dim=0).unsqueeze(0)   # (1, D, H, W)
        mask_vol = torch.stack(masks, dim=0)              # (D, H, W)
        
        # apply 3D transforms
        if self.image_transform is not None:
            img_vol = self.image_transform(img_vol)       # keep float
        if self.mask_transform is not None:
            mask_vol = self.mask_transform(mask_vol.unsqueeze(0)).squeeze(0).long()

        return img_vol.float(), mask_vol.long()


# Transforms (3D)
image_transform = Compose([
    Resize((112, 224, 224)),
    RandFlip(prob=0.5, spatial_axis=0),
    RandAffine(prob=0.3, rotate_range=(0.1, 0.1, 0.1), translate_range=(5,5,5), scale_range=(0.1, 0.1, 0.1)),
    ScaleIntensity()
])

mask_transform = Compose([
    Resize((112, 224, 224), mode="nearest"),
    RandFlip(prob=0.5, spatial_axis=0),
    RandAffine(prob=0.3, rotate_range=(0.1, 0.1, 0.1), translate_range=(5,5,5), scale_range=(0.1, 0.1, 0.1), mode="nearest"),
])


# Model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=5,         
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2), 
    num_res_units=2,
).to(DEVICE)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Data splits and loaders
def list_patient_ids(image_dir: str) -> List[str]:
    return [pid for pid in os.listdir(image_dir)
            if os.path.isdir(os.path.join(image_dir, pid))]

patient_ids = list_patient_ids(IMAGE_DIR)
print(f"Total unique patients: {len(patient_ids)}")

train_ids, temp_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
val_ids, test_ids   = train_test_split(temp_ids, test_size=0.5, random_state=42)
print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")

train_ds = KneeVolumeDataset3D(IMAGE_DIR, MASK_DIR, train_ids, image_transform, mask_transform)
val_ds   = KneeVolumeDataset3D(IMAGE_DIR, MASK_DIR, val_ids,   image_transform, mask_transform)
test_ds  = KneeVolumeDataset3D(IMAGE_DIR, MASK_DIR, test_ids,  image_transform, mask_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, drop_last=False)

print(f"Test set volumes: {len(test_loader.dataset)}")


# Metrics
# Dice Score Function
def dice_score(pred, target, num_classes=5):
    dices = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        dices.append(dice.item())
    return np.mean(dices)

# IoU Score Function
def iou_score(pred, target, num_classes=5):
    ious = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union != 0 else 1.0)
    return np.mean(ious)


# Train / Val
def train_model(epochs=EPOCHS, save_path=BEST_MODEL_PATH):
    best_iou = 0.0
    all_epoch_ious = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        print(f"\n Epoch {epoch+1} started...")
        for x, y in tqdm(train_loader):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(x)
            loss  = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))
        print(f" [Epoch {epoch+1}] Avg Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_ious = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                out  = model(x)
                pred = torch.argmax(out, dim=1)
                for p, t in zip(pred, y):
                    val_ious.append(iou_score(p.cpu(), t.cpu()))
        mean_iou = float(np.mean(val_ious))
        print(f"[Epoch {epoch+1}] Validation IoU: {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at Epoch {epoch+1} with IoU={mean_iou:.4f}")

        all_epoch_ious.append(mean_iou)

    print(f"\n Training finished. Best IoU: {best_iou:.4f}")
    print(f" Mean Validation IoU across all epochs: {np.mean(all_epoch_ious):.4f}")


# Test
# Load Best Model
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
model.eval()

# Run Test 
def test_model(test_loader):
    test_ious = []
    test_dices = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            for p, t in zip(pred, y):
                test_ious.append(iou_score(p.cpu(), t.cpu()))
                test_dices.append(dice_score(p.cpu(), t.cpu()))

    mean_iou = np.mean(test_ious)
    mean_dice = np.mean(test_dices)
    print(f"Test IoU:   {mean_iou:.4f}")
    print(f"Test Dice:  {mean_dice:.4f}")



# Run
if __name__ == "__main__":
    train_model(epochs=EPOCHS, save_path=BEST_MODEL_PATH)
    test_model(test_loader)
