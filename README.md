# 🌾 A Multimodal Chain Network (MMCoNet)
一种多模态链网络（MMCoNet）

---

## 📘 项目概述 | Project Overview

本项目提出了一种**多模态链网络（MMCoNet, A Multimodal Chain Network）**，  
用于从**叶片生理参数（SPAD）**到**冠层结构参数（LAI）**的跨尺度估算。

This project introduces **MMCoNet**, a **Multimodal Chain Network** designed to estimate crop physiological–structural traits across scales — from **leaf-level SPAD (chlorophyll content)** to **canopy-level LAI (Leaf Area Index)**.

MMCoNet 包含两个核心阶段：  
MMCoNet consists of two main stages:

- **阶段 1：CoCa-CP（Cross-modal Contrastive Prompt）模型** — 面向 SPAD 反演  
  *Stage 1: CoCa-CP model for SPAD inversion*

- **阶段 2：MMBT-SA（Multimodal BERT with S2MAE and Adaptive-Weight）模型** — 面向 LAI 反演  
  *Stage 2: MMBT-SA model for LAI inversion*

Together, these stages establish an interpretable pipeline bridging **physiological indicators** and **structural parameters** of crops.

---

## 📂 文件结构 | Directory Structure

├── model_crome_promptmulti.py # CoCa-CP 模型结构 | Model architecture (SPAD)

├── train_crome_promptmulti.py # CoCa-CP 训练脚本 | Training script (SPAD)

├── dataset_spad.py # SPAD 阶段数据集定义 | Dataset for SPAD

├── mmbt_model_spad_s2mae_awt.py # MMBT-SA 模型结构 | Model architecture (LAI)

├── train_mmbt_spad_s2mae_awt_multi.py # MMBT-SA 训练脚本 | Training script (LAI)

├── dataset_spad_awt.py # AWT 阶段数据集定义 | Dataset for multi-view stage


│
└── README.md
---


## 🧩 模型结构 | Model Architecture

### 🔹 阶段 1：CoCa-CP（SPAD 反演）
**Cross-modal Contrastive Prompt Model**

**输入 Inputs**：光谱向量、RGB 图像、灰度光谱图像。  
**核心模块 Key Modules**：

- `PromptBlock` — 可学习任务提示向量 (Learnable prompt tokens)  
- `CrossModalAdapter` — 跨模态多头注意力对齐 (Cross-modal multi-head attention)  
- `CoCaEncoder` — 双分支 Transformer 编码器  
- `Contrastive Loss` — 光谱–图像对比约束  

**输出 Outputs**：  
- SPAD 预测值 (SPAD prediction)  

---

### 🔹 阶段 2：MMBT-SA（LAI 反演）
**Multimodal BERT with S2MAE and Adaptive-Weight**

**输入 Inputs**：光谱向量 + SPAD 值 + RGB 双视图 + 灰度双视图。  
**核心模块 Key Modules**：

- `SpectralSPADEncoder` — 基于 S2MAE 的光谱–SPAD 编码器 (S2MAE-initialized spectral encoder)  
- `ViewWeightNet (AWT)` — 多视图自适应加权融合 (Adaptive Weighted Token fusion)  
- `BERT-style Transformer` — 模态融合与对齐 (Cross-modal fusion via Transformer)  
- `Regression Head` — 输出 LAI 估计 (Regression for LAI)

**创新 Innovations**：  
- 结合 S2MAE 预训练与 Transformer 融合 (S2MAE-based pretraining + Transformer fusion)  
- 自适应加权机制自动学习视图重要性 (Adaptive weighting of multiple views)  
- 显式模态嵌入提升可解释性 (Explicit modality embeddings improve interpretability)

---

## 📊 数据集说明 | Dataset Description

### 🧾 `dataset_spad.py` — SPAD 阶段
- 读取 `shuju.xlsx` 中的光谱 + SPAD 列；  
- 图像来自 `photo/` (RGB) 与 `photo2/` (灰度光谱)；  
- 自动执行尺寸调整、翻转、归一化。  

