from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.v2 as T
import random
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import csv
import pickle


# Set random seeds for reproducibility
SEED = 65
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Paths and constants
IMAGE_DIR = "/data_vault/hexai/OAICartilage/image_manual_crops"
MASK_DIR = "/data_vault/hexai/OAICartilage/cropped_annotations_numpy"
SPLIT_PATH = "/home/feg48/2.5D_seg/split_data_final/knee_split_256_noise_notebook.pkl"

MODEL_PTH = "/home/feg48/2.5D_seg/final_results/best_model_256_noise_notebook_5_0.15.pth"
MODEL_CKPT = "/home/feg48/2.5D_seg/final_results/best_model_256_noise_notebook_5_0.15.ckpt"
TRAIN_LOG_CSV = "/home/feg48/2.5D_seg/final_results/256_noise_training_log_5_0.15.csv"
TRAIN_META_PKL = "/home/feg48/2.5D_seg/final_results/256_training_metadata_5_0.15.pkl"

TEST_SAVE_DIR = "/home/feg48/2.5D_seg/final_results/256_probs_5_0.15"


NUM_CLASSES = 5
INPUT_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 30

# Transforms
# Train/val has a light Gaussian blur augmentation.
train_val_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE)),
        transforms.ToTensor(),
        T.GaussianBlur(5, sigma=0.15),
    ])

test_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE)),
        transforms.ToTensor(),
    ])

label_transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
    ])

# Dataset
class KneeSegmentation25D(Dataset):
    def __init__(self, image_dir, mask_dir, filenames, img_transforms, label_transforms):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = sorted(filenames)
        self.img_transforms = img_transforms
        self.label_transforms = label_transforms


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        study_id, slice_tag = filename.replace(".jpg", "").split("_slice_")
        slice_num = int(slice_tag)
        
        # Build 3-slice stack: [t-1, t, t+1]
        stack = []
        for offset in [-1, 0, 1]:
            n = slice_num + offset
            neighbor_file = f"{study_id}_slice_{n:03d}.jpg"
            neighbor_path = os.path.join(self.image_dir, neighbor_file)
            if os.path.exists(neighbor_path):
                img = Image.open(neighbor_path).convert("L")
            else:
                img = Image.open(os.path.join(self.image_dir, filename)).convert("L")

            img = self.img_transforms(img)
            img = img.squeeze(0) 
            stack.append(img)
            
        image = np.stack(stack, axis=0)  

        mask_path = os.path.join(self.mask_dir, study_id, filename.replace(".jpg", ".npy"))
        mask = np.load(mask_path).astype(np.int64)
        mask = Image.fromarray(mask.astype(np.uint8))
        mask_resized = self.label_transforms(mask).squeeze(0).long()
        
        return image, mask_resized

# Metrics
# dice_score: macro over classes 1..NUM_CLASSES-1 (ignores background=0), 
# iou_score: macro IoU over the same set
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

def iou_score(pred, target, num_classes=5):
    ious = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        ious.append(intersection / union if union != 0 else 1.0)
    return np.mean(ious)

# Model, loss, optimizer
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=5
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Load the saved split
with open(SPLIT_PATH, "rb") as f:
    split_dict = pickle.load(f)

train_f = split_dict["train"]
val_f = split_dict["val"]
test_f = split_dict["test"]

print(f"Loaded split from {SPLIT_PATH}")
print(f"Train: {len(train_f)} | Val: {len(val_f)} | Test: {len(test_f)}")

# Datasets and loaders
train_ds = KneeSegmentation25D(IMAGE_DIR, MASK_DIR, train_f, train_val_transform, label_transform)
val_ds = KneeSegmentation25D(IMAGE_DIR, MASK_DIR, val_f, train_val_transform, label_transform)
test_ds = KneeSegmentation25D(IMAGE_DIR, MASK_DIR, test_f, test_transform, label_transform)


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)


# Training and validation
def train_model(
epochs: int = EPOCHS,
    pth_path: str = MODEL_PTH,
    ckpt_path: str = MODEL_CKPT,
    log_csv_path: str = TRAIN_LOG_CSV,
    pkl_save_path: str = TRAIN_META_PKL
): 

    """
    Trains the model and saves:
      - best .pth (state_dict) by val IoU
      - checkpoint with optimizer and epoch
      - CSV log of per-epoch metrics
      - PKL with training metadata
    """
    best_iou = 0.0
    metrics_log = []  

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        train_dice = []   

        print(f"\n Epoch {epoch+1} started...")

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred_labels = torch.argmax(preds, dim=1)
            for p, t in zip(pred_labels, y):
                train_dice.append(dice_score(p.cpu(), t.cpu()))

        avg_train_loss = total_loss / len(train_loader)
        mean_train_dice = np.mean(train_dice)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train IoU: {mean_train_dice:.4f}")
 


        # Validation
        model.eval()
        val_loss = 0
        val_dice = []
        val_iou = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                val_loss += loss.item()   
                pred = torch.argmax(out, dim=1)
                # print("Prediction shape:", preds.shape)
                # break  # test on just one batch for now
                for p, t in zip(pred, y):
                    val_dice.append(dice_score(p.cpu(), t.cpu()))
                    val_iou.append(iou_score(p.cpu(), t.cpu()))

        avg_val_loss = val_loss / len(val_loader)   
        mean_val_dice = np.mean(val_dice)
        mean_val_iou = np.mean(val_iou)
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Val Dice: {mean_val_dice:.4f}, Val IoU: {mean_val_iou:.4f}")

        # Save best model
        if mean_val_iou > best_iou:
            best_iou = mean_val_iou
            torch.save(model.state_dict(), pth_path)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
            }, ckpt_path)
            print(f"Saved best model at Epoch {epoch+1} with IoU={mean_val_iou:.4f}")

        # Append metrics for CSV logging
        metrics_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_dice": mean_train_dice,
            "val_loss": avg_val_loss,
            "val_iou": mean_val_iou,
            "val_dice": mean_val_dice
        })

    # Training summary
    print(f"\nTraining finished. Best IoU: {best_iou:.4f}")

    # Save logs to CSV
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    with open(log_csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_dice", "val_loss", "val_iou", "val_dice"])
        writer.writeheader()
        for row in metrics_log:
            writer.writerow(row)
    print(f"Training metrics saved to: {log_csv_path}")

    # Save metadata PKL
    with open(pkl_save_path, "wb") as f:
        pickle.dump({
            "best_iou": best_iou,
            "epochs_ran": epochs,
            "metrics_log": metrics_log
        }, f)


    print(f"Training metadata saved to: {pkl_save_path}")

# Run training
train_model(epochs=EPOCHS)


# Test
# Load best and test
model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
model.eval()


# Run Test 
def test_model_and_save_probs(test_loader, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    test_ious = []
    test_dices = []

    with torch.no_grad():
        for idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)

            out = model(x)  # raw logits
            prob = F.softmax(out, dim=1)  # softmax probabilities [B, C, H, W]

            pred = torch.argmax(prob, dim=1)

            # Save per-sample softmax probabilities (for fusion)
            for i in range(x.size(0)):
                sample_prob = prob[i].cpu().numpy()  # shape [C, H, W]
                save_path = os.path.join(save_dir, f"prob_{idx * test_loader.batch_size + i:04d}.npy")
                np.save(save_path, sample_prob)

                p = pred[i].cpu()
                t = y[i].cpu()
                test_ious.append(iou_score(p, t))
                test_dices.append(dice_score(p, t))

    mean_iou = np.mean(test_ious)
    mean_dice = np.mean(test_dices)
    print(f"Test IoU:   {mean_iou:.4f}")
    print(f"Test Dice:  {mean_dice:.4f}")
    print(f"Saved softmax probs to: {save_dir}")

test_model_and_save_probs(test_loader, model, TEST_SAVE_DIR)
