import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class LAIDatasetAWT(Dataset):
    """
    与原 LAIDataset 基本一致，但为 RGB / 光谱灰度图各生成 2 个视图（v1, v2）。
    - Excel 列同原版：
        col0: name
        col1: LAI (label)
        col2: SPAD
        col3 ~ col3+27-1: 27列光谱
        后续列为植被指数（可为0列）
    - 图像路径与存在性检查同原版
    - 变换：为多样性，视图1与视图2使用“独立的随机增强管线”
    """
    def __init__(self, root, excel_name="shuju.xlsx",
                 photo_dir="photo", hyperspec_dir="photo2",
                 rgb_size=224, hyperspec_size=224,
                 rgb_aug_strength=0.4, hs_aug_strength=0.3):
        super().__init__()
        self.root = root
        df = pd.read_excel(os.path.join(root, excel_name))
        self.names = df.iloc[:, 0].tolist()
        self.lai = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32)
        self.spad = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)
        wave_start = 3
        wave_end = wave_start + 27
        spectra = torch.tensor(df.iloc[:, wave_start:wave_end].values, dtype=torch.float32)
        vi_cols = df.iloc[:, wave_end:]
        indices = torch.tensor(vi_cols.values, dtype=torch.float32) if vi_cols.shape[1] > 0 else torch.zeros((len(df), 0))
        self.spec_full = torch.cat([spectra, indices], dim=1)

        # 路径
        img_root = os.path.join(root, photo_dir)
        self.rgb_paths = [os.path.join(img_root, f"{n}.png") for n in self.names]
        for p in self.rgb_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"RGB图像文件 {p} 不存在")

        hs_root = os.path.join(root, hyperspec_dir)
        self.hyperspec_paths = [os.path.join(hs_root, f"{n}.jpg") for n in self.names]
        for p in self.hyperspec_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"光谱图像文件 {p} 不存在")

        # ========== 两套独立的随机增强（v1 / v2）==========
        # 注：参数尽量温和，避免破坏LAI相关结构；你可按需要调大/调小
        self.rgb_tf_v1 = T.Compose([
            T.Resize((rgb_size, rgb_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.ColorJitter(brightness=rgb_aug_strength,
                                         contrast=rgb_aug_strength,
                                         saturation=rgb_aug_strength,
                                         hue=0.1)], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.rgb_tf_v2 = T.Compose([
            T.Resize((rgb_size, rgb_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.RandomApply([T.ColorJitter(brightness=rgb_aug_strength*0.7,
                                         contrast=rgb_aug_strength*0.7,
                                         saturation=rgb_aug_strength*0.7,
                                         hue=0.05)], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        self.hs_tf_v1 = T.Compose([
            T.Resize((hyperspec_size, hyperspec_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        self.hs_tf_v2 = T.Compose([
            T.Resize((hyperspec_size, hyperspec_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        spec_vec = self.spec_full[idx]                   # (D,)
        spad_val = self.spad[idx]                        # ()

        # RGB 两视图
        rgb_img = Image.open(self.rgb_paths[idx]).convert("RGB")
        rgb_v1 = self.rgb_tf_v1(rgb_img)
        rgb_v2 = self.rgb_tf_v2(rgb_img)

        # 灰度两视图
        hs_img = Image.open(self.hyperspec_paths[idx]).convert("L")
        hs_v1 = self.hs_tf_v1(hs_img)
        hs_v2 = self.hs_tf_v2(hs_img)

        return {
            "spectral": spec_vec,
            "spad": spad_val,
            "rgb_v1": rgb_v1,
            "rgb_v2": rgb_v2,
            "hs_v1": hs_v1,
            "hs_v2": hs_v2,
            "lai": self.lai[idx],
        }
