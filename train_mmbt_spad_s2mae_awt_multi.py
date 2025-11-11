import os, math, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, random_split
from dataset_spad_awt import LAIDatasetAWT
from mmbt_model_spad_s2mae_awt import MMBTModel_S2MAE_AWT
from torch.cuda.amp import GradScaler, autocast

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BATCH_SIZE = 4
LR = 1e-4
NUM_EPOCHS = 500
NUM_WORKERS = 8
D_MODEL = 768
NUM_LAYERS = 4
DROPOUT = 0.1
FREEZE_CNN = False
FREEZE_S2MAE = False
DATA_DIR = "/home/bxz/woodsun/HN/DP/data"
S2MAE_CKPT_ROOT = "./pretrain_s2mae_ckpt_multi"    
RESULTS_DIR_NAME = "experiment_records_mmbt_spad_s2mae_awt_multi"
SEED = 42
USE_AMP = True
ACCUM_STEPS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    sq, ab, ape, n = 0., 0., 0., 0
    ys, ps = [], []
    inference_time = 0.0
    for b in loader:
        s = b["spectral"].to(device)
        spad = b["spad"].to(device)
        rgb_v1 = b["rgb_v1"].to(device)
        rgb_v2 = b["rgb_v2"].to(device)
        hs_v1 = b["hs_v1"].to(device)
        hs_v2 = b["hs_v2"].to(device)
        y = b["lai"].to(device)
        t0 = time.time()
        with autocast(enabled=USE_AMP):
            p, _ = net(s, spad, rgb_v1, rgb_v2, hs_v1, hs_v2)
            p = p.squeeze(1)
        inference_time += (time.time() - t0)
        d = p - y
        sq += torch.sum(d**2).item()
        ab += torch.sum(torch.abs(d)).item()
        ape += torch.sum(torch.abs(d)/(y+1e-8)).item()
        n += y.numel()
        ys.append(y); ps.append(p)
    ys = torch.cat(ys); ps = torch.cat(ps)
    mse = sq / n; rmse = math.sqrt(mse); mae = ab / n; mape = ape / n * 100
    r2 = 1 - sq / torch.sum((ys - ys.mean())**2).item()
    pearson_r = torch.corrcoef(torch.stack([ys, ps]))[0,1].item() if n > 1 else 0.0
    return mse, rmse, mae, mape, r2, pearson_r, inference_time / max(1, len(loader))

def save_logs(out_dir, losses, metrics, extra_logs=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write("Epoch\tLoss\tMSE\tRMSE\tMAE\tMAPE\tR2\tPearson_r\tInference_time\n")
        for ep, (l, mse, rm, ma, mp, r2, pr, it) in enumerate(
            zip(losses, metrics['mse'], metrics['rmse'], metrics['mae'], metrics['mape'],
                metrics['r2'], metrics['pearson_r'], metrics['inference_time']), 1):
            f.write(f"{ep}\t{l:.4f}\t{mse:.4f}\t{rm:.4f}\t{ma:.4f}\t{mp:.2f}%\t{r2:.4f}\t{pr:.4f}\t{it:.4f}\n")
    df = pd.DataFrame({
        "Epoch": range(1, len(losses)+1),
        "Loss": losses,
        "MSE": metrics['mse'],
        "RMSE": metrics['rmse'],
        "MAE": metrics['mae'],
        "MAPE": metrics['mape'],
        "R2": metrics['r2'],
        "Pearson_r": metrics['pearson_r'],
        "Inference_time": metrics['inference_time']
    })
    df.to_excel(os.path.join(out_dir, "training_log.xlsx"), index=False)
    plt.figure(figsize=(8,5)); plt.plot(losses); plt.title("Train Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.savefig(os.path.join(out_dir, "train_loss.png")); plt.close()
    plt.figure(figsize=(12,6))
    for k in ['mse', 'rmse', 'mae', 'mape', 'r2', 'pearson_r']:
        plt.plot(metrics[k], label=k.upper())
    plt.legend(); plt.title("Validation Metrics"); plt.xlabel("Epoch")
    plt.savefig(os.path.join(out_dir, "val_metrics.png")); plt.close()

    if extra_logs is not None:
        wlog_path = os.path.join(out_dir, "view_weights_avg.txt")
        with open(wlog_path, "w", encoding="utf-8") as f:
            f.write("Epoch\trgb_w1\trgb_w2\ths_w1\ths_w2\n")
            for i, row in enumerate(extra_logs, 1):
                f.write(f"{i}\t{row['rgb'][0]:.4f}\t{row['rgb'][1]:.4f}\t{row['hs'][0]:.4f}\t{row['hs'][1]:.4f}\n")

def save_hparams(out_dir, params: dict):
    with open(os.path.join(out_dir, "hyperparameters.txt"), "w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

def train_on_subfolder(sub_root, sub):
    set_seed(SEED)
    out_dir = os.path.join(DATA_DIR, RESULTS_DIR_NAME, sub)
    os.makedirs(out_dir, exist_ok=True)

    s2mae_path = os.path.join(S2MAE_CKPT_ROOT, sub, "spectral_mae_best.pth")
    if not os.path.exists(s2mae_path):
        print(f" Warning: No S2MAE ckpt for {sub}, use random init.")
        s2mae_path = None

    try:
        full = LAIDatasetAWT(root=sub_root, photo_dir="photo", hyperspec_dir="photo2")
        print(f"Subfolder {sub} dataset size: {len(full)}")
    except FileNotFoundError as e:
        print(f" Error in {sub}: {e}")
        return None

    spec_dim = full[0]["spectral"].numel()
    tr_len = int(0.8 * len(full)); va_len = len(full) - tr_len
    print(f"Subfolder {sub} - Train size: {tr_len}, Validation size: {va_len}")
    bs = min(BATCH_SIZE, tr_len)
    if bs < BATCH_SIZE:
        print(f"Warning: Train size {tr_len} < BATCH_SIZE {BATCH_SIZE}. Using {bs}.")
    train_ds, val_ds = random_split(full, [tr_len, va_len], generator=torch.Generator().manual_seed(SEED))
    loader_kw = dict(batch_size=bs, num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # 模型
    net = MMBTModel_S2MAE_AWT(
        num_spec_feats=spec_dim, d_model=D_MODEL, num_layers=NUM_LAYERS, dropout=DROPOUT,
        freeze_cnn=FREEZE_CNN, pretrained_s2mae_path=s2mae_path, freeze_s2mae=FREEZE_S2MAE
    ).to(device)

    # 训练准备
    opt = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=USE_AMP)
    losses, metrics = [], {k: [] for k in ['mse','rmse','mae','mape','r2','pearson_r','inference_time']}
    view_weight_logs = []

    # 训练循环
    for epoch in range(1, NUM_EPOCHS + 1):
        net.train()
        epoch_loss = 0.0
        opt.zero_grad(set_to_none=True)

        rgb_w_list, hs_w_list = [], []

        for i, b in enumerate(train_loader):
            s = b["spectral"].to(device)
            spad = b["spad"].to(device)
            rgb_v1 = b["rgb_v1"].to(device)
            rgb_v2 = b["rgb_v2"].to(device)
            hs_v1 = b["hs_v1"].to(device)
            hs_v2 = b["hs_v2"].to(device)
            y = b["lai"].to(device)

            with autocast(enabled=USE_AMP):
                pred, weights = net(s, spad, rgb_v1, rgb_v2, hs_v1, hs_v2)
                pred = pred.squeeze(1)
                loss = criterion(pred, y) / ACCUM_STEPS

            scaler.scale(loss).backward()
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            epoch_loss += loss.item() * ACCUM_STEPS

            rgb_w_list.append(weights["rgb_w"].detach().mean(dim=0).cpu().numpy())  # (2,)
            hs_w_list.append(weights["hs_w"].detach().mean(dim=0).cpu().numpy())    # (2,)

        epoch_loss /= max(1, len(train_loader))
        losses.append(epoch_loss)

        # 记录权重均值
        rgb_w_avg = np.mean(np.stack(rgb_w_list), axis=0) if len(rgb_w_list) else np.array([0.5,0.5])
        hs_w_avg  = np.mean(np.stack(hs_w_list), axis=0) if len(hs_w_list) else np.array([0.5,0.5])
        view_weight_logs.append({"rgb": rgb_w_avg, "hs": hs_w_avg})

        mse, rmse, mae, mape, r2, pr, it = evaluate(net, val_loader)
        for k, v in zip(['mse','rmse','mae','mape','r2','pearson_r','inference_time'], [mse, rmse, mae, mape, r2, pr, it]):
            metrics[k].append(v)

        print(f"[{sub}] Epoch {epoch:03d}/{NUM_EPOCHS} | Loss {epoch_loss:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f} "
              f"| rgb_w={rgb_w_avg} | hs_w={hs_w_avg}")

        save_logs(out_dir, losses, metrics, extra_logs=view_weight_logs)

    # 保存超参
    save_hparams(out_dir, {
        "subfolder": sub,
        "s2mae_path": s2mae_path,
        "freeze_s2mae": FREEZE_S2MAE,
        "freeze_cnn": FREEZE_CNN,
        "epochs": NUM_EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "seed": SEED
    })

    final_metrics = {'mape': metrics['mape'][-1], 'r2': metrics['r2'][-1]}
    print(f" Finished {sub}: MAPE={final_metrics['mape']:.2f}%  R2={final_metrics['r2']:.4f}")
    return final_metrics

def main():
    subfolders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f)) and f.isdigit()]
    subfolders.sort()
    results = {}
    for sub in subfolders:
        sub_root = os.path.join(DATA_DIR, sub)
        try:
            res = train_on_subfolder(sub_root, sub)
            if res is not None:
                results[sub] = res
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f" OOM in {sub}. 尝试提高 ACCUM_STEPS 或设 FREEZE_CNN=True")
            else:
                print(f" Error in {sub}: {e}")

    if results:
        min_mape_sub = min(results, key=lambda k: results[k]['mape'])
        max_r2_sub = max(results, key=lambda k: results[k]['r2'])
        print("\n======= Summary =======")
        print(f"最小 MAPE: {results[min_mape_sub]['mape']:.2f}% @ {min_mape_sub}")
        print(f"最大 R2  : {results[max_r2_sub]['r2']:.4f} @ {max_r2_sub}")
    else:
        print("No subfolder finished successfully.")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
