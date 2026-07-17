import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class LAIDataset(Dataset):
    """
    读取 shuju.xlsx + photo/ 下 RGB 图像 + photo2/ 下光谱相机图像
    列约定：
        col0: name
        col1: LAI（屏蔽，不加载）
        col2: SPAD
        col3 ~ col3+27-1: 27列为光谱
        其后为植被指数（列数可变，允许为0）
    图像：
        photo/: RGB图像，PNG格式，640x360，24位
        photo2/: 光谱相机图像，JPG格式，1600x1200，8位（单通道）
    """
    def __init__(self, root="/home/bxz/woodsun/HN/DP", 
                 excel_name="shuju.xlsx", photo_dir="photo", hyperspec_dir="photo2", 
                 rgb_size=224, hyperspec_size=224):
        super().__init__()
        self.root = root
        df = pd.read_excel(os.path.join(root, excel_name))
        self.names = df.iloc[:, 0].tolist()
        self.spad = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32)  # 只加载 SPAD
        wave_start = 3
        wave_end = wave_start + 27
        spectra = torch.tensor(df.iloc[:, wave_start:wave_end].values, dtype=torch.float32)
        vi_cols = df.iloc[:, wave_end:]
        indices = torch.tensor(vi_cols.values, dtype=torch.float32) if vi_cols.shape[1] > 0 else torch.zeros((len(df),0))
        self.spec_full = torch.cat([spectra, indices], dim=1)
        img_root = os.path.join(root, photo_dir)
        self.rgb_paths = [os.path.join(img_root, f"{n}.png") for n in self.names]
        for p in self.rgb_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"RGB图像文件 {p} 不存在")
        self.rgb_transform = T.Compose([
            T.Resize((rgb_size, rgb_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        hyperspec_root = os.path.join(root, hyperspec_dir)
        self.hyperspec_paths = [os.path.join(hyperspec_root, f"{n}.jpg") for n in self.names]
        for p in self.hyperspec_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"光谱图像文件 {p} 不存在")
        self.hyperspec_transform = T.Compose([
            T.Resize((hyperspec_size, hyperspec_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        spec_vec = self.spec_full[idx]
        rgb_img = Image.open(self.rgb_paths[idx]).convert("RGB")
        rgb_img = self.rgb_transform(rgb_img)
        hyperspec_img = Image.open(self.hyperspec_paths[idx]).convert("L")
        hyperspec_img = self.hyperspec_transform(hyperspec_img)
        return {
            "spectral": spec_vec,
            "rgb_image": rgb_img,
            "hyperspec_image": hyperspec_img,
            "spad": self.spad[idx]  # 只返回 SPAD 作为标签
        }