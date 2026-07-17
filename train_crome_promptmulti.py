import os
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from dataset_spad import LAIDataset as SPADDataset
from model_crome_promptmulti import ViLTCoCaModel_CROME_PromptMulti


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BATCH_SIZE = 4
LR = 1e-4
NUM_EPOCHS = 500
NUM_WORKERS = 8
D_MODEL = 256
FREEZE_CNN = False
DATA_DIR = "/home/bxz/woodsun/HN/SPAD/data"
RESULTS_DIR_NAME = "records_crome_promptmulti"
SEED = 42
USE_AMP = True
ACCUM_STEPS = 4
LAMBDA_REG = 1.0
LAMBDA_CONTRAST = 0.3

# 数据集划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    sq, ab, ape, n = 0.0, 0.0, 0.0, 0
    ys, ps = [], []

    for batch in loader:
        spectral = batch["spectral"].to(device)
        rgb = batch["rgb_image"].to(device)
        gray = batch["hyperspec_image"].to(device)
        target = batch["spad"].to(device)

        with autocast(enabled=USE_AMP):
            pred, _, _ = net(spectral, rgb, gray)

        diff = pred - target
        sq += torch.sum(diff ** 2).item()
        ab += torch.sum(torch.abs(diff)).item()
        ape += torch.sum(torch.abs(diff) / (torch.abs(target) + 1e-8)).item()
        n += target.numel()
        ys.append(target.detach().cpu())
        ps.append(pred.detach().cpu())

    if n == 0:
        raise ValueError("评价数据集为空，无法计算指标。")

    ys = torch.cat(ys)
    ps = torch.cat(ps)
    mse = sq / n
    rmse = math.sqrt(mse)
    mae = ab / n
    mape = ape / n * 100

    denominator = torch.sum((ys - ys.mean()) ** 2).item()
    r2 = 1 - sq / denominator if denominator > 0 else float("nan")
    pearson_r = (
        torch.corrcoef(torch.stack([ys, ps]))[0, 1].item()
        if n > 1
        else float("nan")
    )
    return mse, rmse, mae, mape, r2, pearson_r


def split_dataset(dataset):
    """按照70%:15%:15%固定随机划分训练、验证和测试子集。"""
    total_len = len(dataset)
    train_len = int(total_len * TRAIN_RATIO)
    val_len = int(total_len * VAL_RATIO)
    test_len = total_len - train_len - val_len

    if train_len <= 0 or val_len <= 0 or test_len <= 0:
        raise ValueError(
            "数据量过少，无法按照70%:15%:15%划分："
            f"total={total_len}, train={train_len}, "
            f"val={val_len}, test={test_len}"
        )

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=generator,
    )
    return train_ds, val_ds, test_ds


def save_validation_log(out_dir, losses, metrics):
    log_df = pd.DataFrame(
        {
            "Epoch": range(1, len(losses) + 1),
            "Train_Loss": losses,
            "Val_MSE": metrics["mse"],
            "Val_RMSE": metrics["rmse"],
            "Val_MAE": metrics["mae"],
            "Val_MAPE": metrics["mape"],
            "Val_R2": metrics["r2"],
            "Val_Pearson_r": metrics["pearson_r"],
        }
    )
    log_df.to_excel(os.path.join(out_dir, "training_log.xlsx"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss.png"))
    plt.close()


def train_one_folder(sub_root, sub_name):
    set_seed(SEED)

    out_dir = os.path.join(DATA_DIR, RESULTS_DIR_NAME, sub_name)
    os.makedirs(out_dir, exist_ok=True)

    dataset = SPADDataset(
        root=sub_root,
        photo_dir="photo",
        hyperspec_dir="photo2",
    )
    train_ds, val_ds, test_ds = split_dataset(dataset)

    print(
        f"[{sub_name}] Dataset split: "
        f"Train={len(train_ds)}, "
        f"Validation={len(val_ds)}, "
        f"Test={len(test_ds)}"
    )

    train_batch_size = min(BATCH_SIZE, len(train_ds))
    if train_batch_size < BATCH_SIZE:
        print(
            f"Warning: Train size {len(train_ds)} < BATCH_SIZE {BATCH_SIZE}. "
            f"Using batch size {train_batch_size}."
        )

    loader_kwargs = {
        "batch_size": train_batch_size,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "drop_last": False,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    spec_dim = dataset[0]["spectral"].numel()
    net = ViLTCoCaModel_CROME_PromptMulti(
        spec_dim,
        d_model=D_MODEL,
        freeze_cnn=FREEZE_CNN,
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=USE_AMP)

    losses = []
    metrics = {
        key: []
        for key in ["mse", "rmse", "mae", "mape", "r2", "pearson_r"]
    }

    best_val_r2 = -float("inf")
    best_epoch = 0
    best_model_path = os.path.join(out_dir, "best_model.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        net.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_index, batch in enumerate(train_loader):
            spectral = batch["spectral"].to(device)
            rgb = batch["rgb_image"].to(device)
            gray = batch["hyperspec_image"].to(device)
            target = batch["spad"].to(device)

            with autocast(enabled=USE_AMP):
                pred, spec_c, img_c = net(spectral, rgb, gray)
                reg_loss = criterion(pred, target)
                contrast = contrastive_loss(spec_c, img_c)
                loss = (
                    LAMBDA_REG * reg_loss
                    + LAMBDA_CONTRAST * contrast
                ) / ACCUM_STEPS

            scaler.scale(loss).backward()

            should_step = (
                (batch_index + 1) % ACCUM_STEPS == 0
                or (batch_index + 1) == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * ACCUM_STEPS

        epoch_loss /= max(1, len(train_loader))
        losses.append(epoch_loss)

        mse, rmse, mae, mape, r2, pearson_r = evaluate(net, val_loader)
        for key, value in zip(
            metrics.keys(),
            [mse, rmse, mae, mape, r2, pearson_r],
        ):
            metrics[key].append(value)

        if not math.isnan(r2) and r2 > best_val_r2:
            best_val_r2 = r2
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "val_r2": r2,
                    "model_state_dict": net.state_dict(),
                    "spec_dim": spec_dim,
                },
                best_model_path,
            )

        print(
            f"[{sub_name}] Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Loss {epoch_loss:.4f} | "
            f"Val RMSE {rmse:.4f} | "
            f"Val MAE {mae:.4f} | "
            f"Val MAPE {mape:.2f}% | "
            f"Val R2 {r2:.4f}"
        )

    save_validation_log(out_dir, losses, metrics)

    if not os.path.exists(best_model_path):
        raise RuntimeError("训练完成后未生成最佳模型文件。")

    checkpoint = torch.load(best_model_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])

    test_mse, test_rmse, test_mae, test_mape, test_r2, test_pr = evaluate(
        net,
        test_loader,
    )

    print(
        f"\n[{sub_name}] Final Test Results "
        f"(best validation model, epoch={best_epoch}):\n"
        f"MSE      = {test_mse:.4f}\n"
        f"RMSE     = {test_rmse:.4f}\n"
        f"MAE      = {test_mae:.4f}\n"
        f"MAPE     = {test_mape:.2f}%\n"
        f"R2       = {test_r2:.4f}\n"
        f"Pearson  = {test_pr:.4f}"
    )

    pd.DataFrame(
        {
            "Metric": [
                "Best_validation_epoch",
                "Best_validation_R2",
                "Test_MSE",
                "Test_RMSE",
                "Test_MAE",
                "Test_MAPE",
                "Test_R2",
                "Test_Pearson_r",
            ],
            "Value": [
                best_epoch,
                best_val_r2,
                test_mse,
                test_rmse,
                test_mae,
                test_mape,
                test_r2,
                test_pr,
            ],
        }
    ).to_excel(os.path.join(out_dir, "test_results.xlsx"), index=False)

    with open(
        os.path.join(out_dir, "dataset_split.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(f"seed: {SEED}\n")
        file.write(f"total: {len(dataset)}\n")
        file.write(f"train: {len(train_ds)}\n")
        file.write(f"validation: {len(val_ds)}\n")
        file.write(f"test: {len(test_ds)}\n")
        file.write("ratio: 70%:15%:15%\n")

    return test_mape, test_r2


def main():
    subfolders = [
        folder
        for folder in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, folder)) and folder.isdigit()
    ]
    subfolders.sort()

    results = {}
    for sub in subfolders:
        print(f"Training on subfolder: {sub}")
        sub_root = os.path.join(DATA_DIR, sub)
        try:
            test_mape, test_r2 = train_one_folder(sub_root, sub)
            results[sub] = {"mape": test_mape, "r2": test_r2}
        except Exception as error:
            print(f"Error in {sub}: {error}")

    if results:
        best_mape = min(results.items(), key=lambda item: item[1]["mape"])
        best_r2 = max(results.items(), key=lambda item: item[1]["r2"])
        print("\n======= Test Summary =======")
        print(
            f"Best Test MAPE: {best_mape[1]['mape']:.2f}% "
            f"in {best_mape[0]}"
        )
        print(
            f"Best Test R2: {best_r2[1]['r2']:.4f} "
            f"in {best_r2[0]}"
        )
    else:
        print("No subfolder finished successfully.")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
