import argparse
import json
import math
import os
import random
import shutil
import subprocess
import uuid
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             classification_report, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# ----------------------------- CONFIG ---------------------------------------
CGT_REPO = "https://github.com/gsalzer/cgt.git"
Forta_DS_HF = "forta/malicious-smart-contract-dataset"
CLASS_NAMES = [
    "Reentrancy",
    "AccessControl",
    "Arithmetic",
    "UncheckedCall",
    "DenialOfService",
    "BadRandomness",
    "FrontRunning",
    "TimeManipulation",
    "ShortAddress",
    "Safe"  # no vulnerability
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_ROOT = "images"
BYTECODE_CACHE = "bytecode_cache"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ------------------------ BYTECODE → IMAGE ----------------------------------

def bytecode2img(bytecode_hex: str, target_size=(256, 256)) -> Image.Image:
    """Convert a hex string of runtime bytecode into a PIL grayscale image."""
    data = bytearray.fromhex(bytecode_hex)
    length = len(data)
    width = int(math.sqrt(length))
    height = math.ceil(length / width)

    # pad to rectangle
    if len(data) < width * height:
        data.extend(b"\x00" * (width * height - len(data)))

    img = Image.frombytes("L", (width, height), bytes(data))
    img = img.resize(target_size, Image.NEAREST)
    return img

# ----------------------- DATA INGESTION -------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clone_cgt(root="cgt_data"):
    if not os.path.isdir(root):
        subprocess.run(["git", "clone", "--depth", "1", CGT_REPO, root], check=True)
    return root


def load_cgt_samples(root, per_class=100):
    """Return list of (source_path, label) for 9 vulnerable classes."""
    meta_csv = os.path.join(root, "metadata.csv")
    if not os.path.isfile(meta_csv):
        raise FileNotFoundError("metadata.csv not found in CGT repo clone; update path")
    meta = pd.read_csv(meta_csv)
    samples = []
    for vul in CLASS_NAMES[:-1]:  # skip safe
        sub = meta[meta["vulnerability"] == vul.lower()]
        if len(sub) < per_class:
            raise ValueError(f"Not enough samples for {vul} in CGT")
        chosen = sub.sample(per_class, random_state=SEED)
        for _, row in chosen.iterrows():
            samples.append((os.path.join(root, row["filepath"]), vul))
    return samples


def load_forta_samples(per_class=100):
    """Download benign contracts list via HuggingFace Hub (text file of solidity).
    Assumes `datasets` library installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        subprocess.run(["pip", "install", "datasets"], check=True)
        from datasets import load_dataset
    ds = load_dataset(Forta_DS_HF, split="train[:100]")
    samples = []
    for i, row in enumerate(ds):
        src = row["source_code"]
        path = f"forta_safe_{i}.sol"
        with open(path, "w") as f:
            f.write(src)
        samples.append((path, "Safe"))
    return samples

# ---------------------- COMPILATION -----------------------------------------

def compile_to_runtime_bytecode(sol_path: str) -> str:
    """Return runtime bytecode (hex string, no 0x) using solcx; cache results."""
    ensure_dir(BYTECODE_CACHE)
    cache_file = os.path.join(BYTECODE_CACHE, os.path.basename(sol_path) + ".json")
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            return json.load(f)["bytecode_runtime"]

    try:
        from solcx import compile_standard, install_solc
    except ImportError:
        subprocess.run(["pip", "install", "py-solc-x"], check=True)
        from solcx import compile_standard, install_solc

    install_solc("0.8.20")
    with open(sol_path, "r") as f:
        source = f.read()

    std_json = {
        "language": "Solidity",
        "sources": {sol_path: {"content": source}},
        "settings": {
            "outputSelection": {"*": {"*": ["evm.bytecode", "evm.deployedBytecode"]}},
        },
    }
    compiled = compile_standard(std_json, solc_version="0.8.20")
    contract_name = next(iter(compiled["contracts"][sol_path]))
    runtime_bc = compiled["contracts"][sol_path][contract_name]["evm"]["deployedBytecode"]["object"]
    with open(cache_file, "w") as f:
        json.dump({"bytecode_runtime": runtime_bc}, f)
    return runtime_bc

# ------------------- IMAGE BUILDING -----------------------------------------

def build_image_dataset(samples):
    ensure_dir(IMG_ROOT)
    for src_path, label in tqdm(samples, desc="Compiling & imaging"):
        try:
            bytecode = compile_to_runtime_bytecode(src_path)
            img = bytecode2img(bytecode)
            out_dir = os.path.join(IMG_ROOT, label)
            ensure_dir(out_dir)
            img.save(os.path.join(out_dir, f"{uuid.uuid4().hex}.png"))
        except Exception as e:
            print(f"[WARN] Failed {src_path}: {e}")

# ------------------- DATASET -------------------------------------------------

def build_dataloaders(batch=32):
    transform = transforms.Compose([
        transforms.ToTensor(),        # [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5]),  # grayscale channel
    ])
    full_ds = torchvision.datasets.ImageFolder(IMG_ROOT, transform=transform)
    targets = np.array(full_ds.targets)

    train_idx, tmp_idx = train_test_split(np.arange(len(full_ds)), test_size=0.2, stratify=targets, random_state=SEED)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.5, stratify=targets[tmp_idx], random_state=SEED)

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=batch, shuffle=True, num_workers=4)
    val_loader   = DataLoader(Subset(full_ds, val_idx),   batch_size=batch, shuffle=False, num_workers=4)
    test_loader  = DataLoader(Subset(full_ds, test_idx),  batch_size=batch, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader, full_ds.class_to_idx

# ------------------- MODEL ---------------------------------------------------

def get_model(num_classes=10):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    # adapt first conv layer to single channel
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ------------------- TRAIN / EVAL -------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        total_correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, total_correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    avg_loss = total_loss / total
    avg_acc = total_correct / total
    return avg_loss, avg_acc, np.concatenate(all_probs), np.concatenate(all_labels)

# ------------------- MAIN ----------------------------------------------------

def main(args):
    # 1. Data acquisition
    cgt_root = clone_cgt()
    samples = load_cgt_samples(cgt_root, per_class=100) + load_forta_samples(100)

    # 2. Bytecode -> images
    build_image_dataset(samples)

    # 3. Dataloaders
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(args.batch)
    print("Class mapping:", class_to_idx)

    # 4. Model, loss, opt
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_acc, best_state = 0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}: train acc {tr_acc:.3f}, val acc {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

    # 5. Testing
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_acc, test_probs, test_labels = eval_epoch(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_acc:.3f}")

    # 6. Metrics
    y_pred = test_probs.argmax(1)
    print(classification_report(test_labels, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(test_labels, y_pred, labels=list(range(len(CLASS_NAMES))))
    np.save("confusion_matrix.npy", cm)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(xticks_rotation=45)
    import matplotlib.pyplot as plt
    plt.tight_layout(); plt.savefig("confusion_matrix.png")

    # ROC-AUC per class (one‑vs‑rest)
    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(test_labels, classes=list(range(len(CLASS_NAMES))))
        aucs = {}
        for i, cls in enumerate(CLASS_NAMES):
            try:
                auc = roc_auc_score(y_bin[:, i], test_probs[:, i])
                aucs[cls] = auc
            except ValueError:
                aucs[cls] = float("nan")
        with open("roc_auc.json", "w") as f:
            json.dump(aucs, f, indent=2)
    except Exception as e:
        print("ROC‑AUC calculation failed:", e)

    torch.save(best_state, "model.pth")
    print("Saved best model to model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
