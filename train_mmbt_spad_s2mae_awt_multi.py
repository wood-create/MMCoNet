import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from dataset_spad_awt import LAIDatasetAWT
from mmbt_model_spad_s2mae_awt import MMBTModel_S2MAE_AWT


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


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    sq, ab, ape, n = 0.0, 0.0, 0.0, 0
    ys, ps = [], []
    inference_time = 0.0

    for batch in loader:
        spectral = batch["spectral"].to(device)
        spad = batch["spad"].to(device)
        rgb_v1 = batch["rgb_v1"].to(device)
        rgb_v2 = batch["rgb_v2"].to(device)
        hs_v1 = batch["hs_v1"].to(device)
        hs_v2 = batch["hs_v2"].to(device)
        target = batch["lai"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        with autocast(enabled=USE_AMP):
            pred, _ = net(
                spectral,
                spad,
                rgb_v1,
                rgb_v2,
                hs_v1,
                hs_v2,
            )
            pred = pred.squeeze(1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        inference_time += time.time() - start_time

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
    average_inference_time = inference_time / max(1, len(loader))

    return mse, rmse, mae, mape, r2, pearson_r, average_inference_time


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


def save_logs(out_dir, losses, metrics, extra_logs=None):
    os.makedirs(out_dir, exist_ok=True)

    with open(
        os.path.join(out_dir, "training_log.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Epoch\tLoss\tMSE\tRMSE\tMAE\tMAPE\tR2\t"
            "Pearson_r\tInference_time\n"
        )
        for epoch, values in enumerate(
            zip(
                losses,
                metrics["mse"],
                metrics["rmse"],
                metrics["mae"],
                metrics["mape"],
                metrics["r2"],
                metrics["pearson_r"],
                metrics["inference_time"],
            ),
            1,
        ):
            loss, mse, rmse, mae, mape, r2, pearson_r, inference = values
            file.write(
                f"{epoch}\t{loss:.4f}\t{mse:.4f}\t{rmse:.4f}\t"
                f"{mae:.4f}\t{mape:.2f}%\t{r2:.4f}\t"
                f"{pearson_r:.4f}\t{inference:.6f}\n"
            )

    pd.DataFrame(
        {
            "Epoch": range(1, len(losses) + 1),
            "Loss": losses,
            "Val_MSE": metrics["mse"],
            "Val_RMSE": metrics["rmse"],
            "Val_MAE": metrics["mae"],
            "Val_MAPE": metrics["mape"],
            "Val_R2": metrics["r2"],
            "Val_Pearson_r": metrics["pearson_r"],
            "Val_Inference_time_per_batch": metrics["inference_time"],
        }
    ).to_excel(os.path.join(out_dir, "training_log.xlsx"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    for key in ["mse", "rmse", "mae", "mape", "r2", "pearson_r"]:
        plt.plot(metrics[key], label=key.upper())
    plt.legend()
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_metrics.png"))
    plt.close()

    if extra_logs is not None:
        with open(
            os.path.join(out_dir, "view_weights_avg.txt"),
            "w",
            encoding="utf-8",
        ) as file:
            file.write("Epoch\trgb_w1\trgb_w2\ths_w1\ths_w2\n")
            for epoch, row in enumerate(extra_logs, 1):
                file.write(
                    f"{epoch}\t{row['rgb'][0]:.4f}\t{row['rgb'][1]:.4f}\t"
                    f"{row['hs'][0]:.4f}\t{row['hs'][1]:.4f}\n"
                )


def save_hparams(out_dir, params):
    with open(
        os.path.join(out_dir, "hyperparameters.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        for key, value in params.items():
            file.write(f"{key}: {value}\n")


def train_on_subfolder(sub_root, sub):
    set_seed(SEED)

    out_dir = os.path.join(DATA_DIR, RESULTS_DIR_NAME, sub)
    os.makedirs(out_dir, exist_ok=True)

    s2mae_path = os.path.join(
        S2MAE_CKPT_ROOT,
        sub,
        "spectral_mae_best.pth",
    )
    if not os.path.exists(s2mae_path):
        print(f"Warning: No S2MAE checkpoint for {sub}; using random initialization.")
        s2mae_path = None

    try:
        full_dataset = LAIDatasetAWT(
            root=sub_root,
            photo_dir="photo",
            hyperspec_dir="photo2",
        )
        print(f"Subfolder {sub} dataset size: {len(full_dataset)}")
    except FileNotFoundError as error:
        print(f"Error in {sub}: {error}")
        return None

    spec_dim = full_dataset[0]["spectral"].numel()
    train_ds, val_ds, test_ds = split_dataset(full_dataset)

    print(
        f"Subfolder {sub} dataset split: "
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

    net = MMBTModel_S2MAE_AWT(
        num_spec_feats=spec_dim,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        freeze_cnn=FREEZE_CNN,
        pretrained_s2mae_path=s2mae_path,
        freeze_s2mae=FREEZE_S2MAE,
    ).to(device)

    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=USE_AMP)

    losses = []
    metrics = {
        key: []
        for key in [
            "mse",
            "rmse",
            "mae",
            "mape",
            "r2",
            "pearson_r",
            "inference_time",
        ]
    }
    view_weight_logs = []

    best_val_r2 = -float("inf")
    best_epoch = 0
    best_model_path = os.path.join(out_dir, "best_model.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        net.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        rgb_weight_list, hs_weight_list = [], []

        for batch_index, batch in enumerate(train_loader):
            spectral = batch["spectral"].to(device)
            spad = batch["spad"].to(device)
            rgb_v1 = batch["rgb_v1"].to(device)
            rgb_v2 = batch["rgb_v2"].to(device)
            hs_v1 = batch["hs_v1"].to(device)
            hs_v2 = batch["hs_v2"].to(device)
            target = batch["lai"].to(device)

            with autocast(enabled=USE_AMP):
                pred, weights = net(
                    spectral,
                    spad,
                    rgb_v1,
                    rgb_v2,
                    hs_v1,
                    hs_v2,
                )
                pred = pred.squeeze(1)
                loss = criterion(pred, target) / ACCUM_STEPS

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
            rgb_weight_list.append(
                weights["rgb_w"].detach().mean(dim=0).cpu().numpy()
            )
            hs_weight_list.append(
                weights["hs_w"].detach().mean(dim=0).cpu().numpy()
            )

        epoch_loss /= max(1, len(train_loader))
        losses.append(epoch_loss)

        rgb_weight_avg = (
            np.mean(np.stack(rgb_weight_list), axis=0)
            if rgb_weight_list
            else np.array([0.5, 0.5])
        )
        hs_weight_avg = (
            np.mean(np.stack(hs_weight_list), axis=0)
            if hs_weight_list
            else np.array([0.5, 0.5])
        )
        view_weight_logs.append(
            {"rgb": rgb_weight_avg, "hs": hs_weight_avg}
        )

        mse, rmse, mae, mape, r2, pearson_r, inference_time = evaluate(
            net,
            val_loader,
        )
        for key, value in zip(
            [
                "mse",
                "rmse",
                "mae",
                "mape",
                "r2",
                "pearson_r",
                "inference_time",
            ],
            [
                mse,
                rmse,
                mae,
                mape,
                r2,
                pearson_r,
                inference_time,
            ],
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
            f"[{sub}] Epoch {epoch:03d}/{NUM_EPOCHS} | "
            f"Loss {epoch_loss:.4f} | "
            f"Val RMSE {rmse:.4f} | "
            f"Val R2 {r2:.4f} | "
            f"rgb_w={rgb_weight_avg} | "
            f"hs_w={hs_weight_avg}"
        )

        save_logs(
            out_dir,
            losses,
            metrics,
            extra_logs=view_weight_logs,
        )

    if not os.path.exists(best_model_path):
        raise RuntimeError("训练完成后未生成最佳模型文件。")

    checkpoint = torch.load(best_model_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])

    test_mse, test_rmse, test_mae, test_mape, test_r2, test_pr, test_time = evaluate(
        net,
        test_loader,
    )

    print(
        f"\n[{sub}] Final Test Results "
        f"(best validation model, epoch={best_epoch}):\n"
        f"MSE              = {test_mse:.4f}\n"
        f"RMSE             = {test_rmse:.4f}\n"
        f"MAE              = {test_mae:.4f}\n"
        f"MAPE             = {test_mape:.2f}%\n"
        f"R2               = {test_r2:.4f}\n"
        f"Pearson          = {test_pr:.4f}\n"
        f"Inference time   = {test_time:.6f} s/batch"
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
                "Test_Inference_time_per_batch",
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
                test_time,
            ],
        }
    ).to_excel(os.path.join(out_dir, "test_results.xlsx"), index=False)

    save_hparams(
        out_dir,
        {
            "subfolder": sub,
            "s2mae_path": s2mae_path,
            "freeze_s2mae": FREEZE_S2MAE,
            "freeze_cnn": FREEZE_CNN,
            "epochs": NUM_EPOCHS,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": ACCUM_STEPS,
            "seed": SEED,
            "split_ratio": "70%:15%:15%",
            "train_samples": len(train_ds),
            "validation_samples": len(val_ds),
            "test_samples": len(test_ds),
            "best_validation_epoch": best_epoch,
            "best_validation_R2": best_val_r2,
        },
    )

    final_metrics = {
        "mape": test_mape,
        "r2": test_r2,
        "mse": test_mse,
        "rmse": test_rmse,
        "mae": test_mae,
        "pearson_r": test_pr,
        "inference_time": test_time,
    }

    print(
        f"Finished {sub}: "
        f"Test MAPE={final_metrics['mape']:.2f}% | "
        f"Test R2={final_metrics['r2']:.4f}"
    )
    return final_metrics


def main():
    subfolders = [
        folder
        for folder in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, folder)) and folder.isdigit()
    ]
    subfolders.sort()

    results = {}
    for sub in subfolders:
        sub_root = os.path.join(DATA_DIR, sub)
        try:
            result = train_on_subfolder(sub_root, sub)
            if result is not None:
                results[sub] = result
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                print(
                    f"OOM in {sub}. 尝试提高ACCUM_STEPS或设置FREEZE_CNN=True。"
                )
            else:
                print(f"Error in {sub}: {error}")
        except Exception as error:
            print(f"Error in {sub}: {error}")

    if results:
        min_mape_sub = min(results, key=lambda key: results[key]["mape"])
        max_r2_sub = max(results, key=lambda key: results[key]["r2"])
        print("\n======= Test Summary =======")
        print(
            f"Best Test MAPE: {results[min_mape_sub]['mape']:.2f}% "
            f"@ {min_mape_sub}"
        )
        print(
            f"Best Test R2: {results[max_r2_sub]['r2']:.4f} "
            f"@ {max_r2_sub}"
        )
    else:
        print("No subfolder finished successfully.")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
