import os
import math
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset_spad import LAIDataset as SPADDataset
from model_crome_promptmulti import ViLTCoCaModel_CROME_PromptMulti
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BATCH_SIZE = 4
LR = 1e-4
NUM_EPOCHS = 500
NUM_WORKERS = 8
D_MODEL = 256
FREEZE_CNN = False
DATA_DIR = "/home/bxz/woodsun/HN/SPAD/data1"
RESULTS_DIR_NAME = "records_crome_promptmulti"
SEED = 42
USE_AMP = True
ACCUM_STEPS = 4
LAMBDA_REG = 1.0
LAMBDA_CONTRAST = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def contrastive_loss(spec_feats, img_feats, temperature=0.07):
    spec_norm = torch.nn.functional.normalize(spec_feats, dim=-1)
    img_norm = torch.nn.functional.normalize(img_feats, dim=-1)
    logits = torch.matmul(spec_norm, img_norm.T) / temperature
    labels = torch.arange(spec_feats.size(0), device=spec_feats.device)
    loss_i2s = torch.nn.functional.cross_entropy(logits, labels)
    loss_s2i = torch.nn.functional.cross_entropy(logits.T, labels)
    return (loss_i2s + loss_s2i) / 2

def evaluate(net, loader):
    net.eval()
    sq, ab, ape, n = 0., 0., 0., 0
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            s = b["spectral"].to(device)
            rgb = b["rgb_image"].to(device)
            gray = b["hyperspec_image"].to(device)
            y = b["spad"].to(device)
            with autocast(enabled=USE_AMP):
                p, _, _ = net(s, rgb, gray)
            d = p - y
            sq += torch.sum(d**2).item()
            ab += torch.sum(torch.abs(d)).item()
            ape += torch.sum(torch.abs(d)/(y + 1e-8)).item()
            n += y.numel()
            ys.append(y)
            ps.append(p)
    ys = torch.cat(ys)
    ps = torch.cat(ps)
    mse = sq / n
    rmse = math.sqrt(mse)
    mae = ab / n
    mape = ape / n * 100
    r2 = 1 - sq / torch.sum((ys - ys.mean())**2).item()
    pr = torch.corrcoef(torch.stack([ys, ps]))[0, 1].item() if n > 1 else 0.0
    return mse, rmse, mae, mape, r2, pr

def train_one_folder(sub_root, sub_name):
    set_seed(SEED)
    out_dir = os.path.join(DATA_DIR, RESULTS_DIR_NAME, sub_name)
    os.makedirs(out_dir, exist_ok=True)
    dataset = SPADDataset(root=sub_root, photo_dir="photo", hyperspec_dir="photo2")
    tr_len = int(0.8 * len(dataset))
    va_len = len(dataset) - tr_len
    train_ds, val_ds = random_split(dataset, [tr_len, va_len], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    spec_dim = dataset[0]["spectral"].numel()
    net = ViLTCoCaModel_CROME_PromptMulti(spec_dim, d_model=D_MODEL, freeze_cnn=FREEZE_CNN).to(device)
    opt = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=USE_AMP)
    losses, metrics = [], {k: [] for k in ['mse', 'rmse', 'mae', 'mape', 'r2', 'pearson_r']}
    for epoch in range(1, NUM_EPOCHS + 1):
        net.train()
        epoch_loss = 0.
        opt.zero_grad()
        for i, b in enumerate(train_loader):
            s = b["spectral"].to(device)
            rgb = b["rgb_image"].to(device)
            gray = b["hyperspec_image"].to(device)
            y = b["spad"].to(device)
            with autocast(enabled=USE_AMP):
                pred, spec_c, img_c = net(s, rgb, gray)
                reg_loss = criterion(pred, y)
                contrast = contrastive_loss(spec_c, img_c)
                loss = (LAMBDA_REG * reg_loss + LAMBDA_CONTRAST * contrast) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            epoch_loss += loss.item() * ACCUM_STEPS
        epoch_loss /= max(1, len(train_loader))
        losses.append(epoch_loss)
        mse, rmse, mae, mape, r2, pr = evaluate(net, val_loader)
        for k, v in zip(metrics.keys(), [mse, rmse, mae, mape, r2, pr]):
            metrics[k].append(v)
        print(f"[{sub_name}] Epoch {epoch:03d} Loss {epoch_loss:.4f} RMSE {rmse:.4f} MAE {mae:.4f} MAPE {mape:.2f}% R2 {r2:.4f}")
    pd.DataFrame(metrics).to_excel(os.path.join(out_dir, "training_log.xlsx"), index=False)
    plt.plot(losses)
    plt.title("Train Loss")
    plt.savefig(os.path.join(out_dir, "train_loss.png"))
    return metrics["mape"][-1], metrics["r2"][-1]

def main():
    subfolders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and f.isdigit()]
    subfolders.sort()
    results = {}
    for sub in subfolders:
        print(f"Training on subfolder: {sub}")
        sub_root = os.path.join(DATA_DIR, sub)
        try:
            mape, r2 = train_one_folder(sub_root, sub)
            results[sub] = {"mape": mape, "r2": r2}
        except Exception as e:
            print(f"Error in {sub}: {e}")
    if results:
        best_mape = min(results.items(), key=lambda x: x[1]["mape"])
        best_r2 = max(results.items(), key=lambda x: x[1]["r2"])
        print(f"\nBest MAPE: {best_mape[1]['mape']:.2f}% in {best_mape[0]}")
        print(f"Best R²: {best_r2[1]['r2']:.4f} in {best_r2[0]}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
