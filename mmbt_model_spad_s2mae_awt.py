import torch
import torch.nn as nn
from torchvision import models
from transformers import BertModel, BertConfig

#CNN
def _load_resnet101_backbone():
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    return nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
    )

def _load_resnet18_backbone_1ch():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
    )

#光谱+SPAD 编码（含S2MAE初始化）
class SpectralSPADEncoder(nn.Module):
    def __init__(self, num_feats: int, d_model=768, pretrained_s2mae_path: str = None, freeze_pretrained: bool = False):
        super().__init__()
        self.num_feats = num_feats
        self.val_proj = nn.Linear(1, d_model)
        self.band_emb = nn.Embedding(num_feats + 1, d_model)  # +1 预留给 SPAD 位置
        self._maybe_load_s2mae(pretrained_s2mae_path, freeze_pretrained)

    def _maybe_load_s2mae(self, path, freeze):
        if not path:
            return
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("state_dict", {})
        if "val_proj" in sd:
            self.val_proj.load_state_dict(sd["val_proj"], strict=True)
        if "band_emb" in sd:
            pre = sd["band_emb"]["weight"]
            with torch.no_grad():
                self.band_emb.weight[:pre.shape[0]].copy_(pre)
        if freeze:
            for p in self.val_proj.parameters():
                p.requires_grad = False
            # 允许 band_emb 的最后一行（SPAD位置）继续训练
            self.band_emb.weight.register_hook(lambda g: g)  # 保持梯度，下面选择性冻结
            for name, p in self.named_parameters():
                if "band_emb" in name:
                    p.requires_grad = True

    def forward(self, spectral_vec, spad):
        B, D = spectral_vec.shape
        assert D == self.num_feats
        val = self.val_proj(spectral_vec.unsqueeze(-1))  # (B, D, d)
        idx = torch.arange(D, device=spectral_vec.device).unsqueeze(0).expand(B, D)
        spec_emb = val + self.band_emb(idx)
        spad_val = self.val_proj(spad.unsqueeze(-1).unsqueeze(-1))  # (B, 1, d)
        spad_idx = torch.full((B, 1), D, device=spad.device, dtype=torch.long)
        spad_emb = spad_val + self.band_emb(spad_idx)               # (B, 1, d)
        return torch.cat([spec_emb, spad_emb], dim=1)               # (B, D+1, d)

#多视图加权网络（AWT: Weight）
class ViewWeightNet(nn.Module):
    """
    输入每个视图的 token（B, 1, d），输出两个视图的权重 w1, w2（softmax），
    然后计算加权和：t = w1*t1 + w2*t2 作为该模态的融合 token。
    """
    def __init__(self, d_model=768, hidden=256):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 1)  # 每个视图 -> 标量分数
        )

    def forward(self, t1, t2):
        # t1, t2: (B, 1, d)
        s1 = self.scorer(t1).squeeze(-1)  # (B, 1)
        s2 = self.scorer(t2).squeeze(-1)  # (B, 1)
        logits = torch.cat([s1, s2], dim=1)  # (B, 2)
        w = torch.softmax(logits, dim=1)     # (B, 2)
        w1 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
        w2 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
        fused = w1 * t1 + w2 * t2            # (B, 1, d)
        return fused, w  # 返回融合token与权重，便于记录/可视化

#主模型（MMBT + S2MAE + AWT-Views&Weight）
class MMBTModel_S2MAE_AWT(nn.Module):
    def __init__(self, num_spec_feats: int, d_model=768, num_layers=4, dropout=0.1,
                 freeze_cnn=False, pretrained_s2mae_path: str = None, freeze_s2mae: bool = False):
        super().__init__()
        self.d_model = d_model

        # 光谱+SPAD（S2MAE初始化）
        self.spec_spad_enc = SpectralSPADEncoder(
            num_feats=num_spec_feats, d_model=d_model,
            pretrained_s2mae_path=pretrained_s2mae_path,
            freeze_pretrained=freeze_s2mae
        )

        # RGB 两视图编码
        self.rgb_backbone = _load_resnet101_backbone()
        self.rgb_head = nn.Linear(2048, d_model)
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rgb_weight = ViewWeightNet(d_model=d_model, hidden=256)

        # 灰度 两视图编码
        self.hs_backbone = _load_resnet18_backbone_1ch()
        self.hs_head = nn.Linear(512, d_model)
        self.hs_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.hs_weight = ViewWeightNet(d_model=d_model, hidden=256)

        if freeze_cnn:
            for p in list(self.rgb_backbone.parameters()) + list(self.hs_backbone.parameters()):
                p.requires_grad = False

        # Transformer 融合（BERT-style）
        config = BertConfig(hidden_size=d_model, num_hidden_layers=num_layers,
                            num_attention_heads=12, intermediate_size=d_model*4,
                            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
        self.transformer = BertModel(config)

        # 模态嵌入（0: spec+spad, 1: rgb, 2: hyperspec）
        self.modality_emb = nn.Embedding(3, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 回归头
        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def _encode_rgb_one(self, x):
        # x: (B,3,H,W)
        f = self.rgb_backbone(x)          # (B,2048,h,w)
        f = self.rgb_pool(f).flatten(1)   # (B,2048)
        return self.rgb_head(f).unsqueeze(1)  # (B,1,d)

    def _encode_hs_one(self, x):
        # x: (B,1,H,W)
        f = self.hs_backbone(x)          # (B,512,h,w)
        f = self.hs_pool(f).flatten(1)   # (B,512)
        return self.hs_head(f).unsqueeze(1)   # (B,1,d)

    def forward(self, spectral_vec, spad, rgb_v1, rgb_v2, hs_v1, hs_v2):
        B = spectral_vec.shape[0]

        # 光谱+SPAD -> 多token
        spec_spad_tokens = self.spec_spad_enc(spectral_vec, spad)     # (B, D+1, d)

        # 两视图编码 -> 加权融合为单token
        rgb_t1 = self._encode_rgb_one(rgb_v1)      # (B,1,d)
        rgb_t2 = self._encode_rgb_one(rgb_v2)      # (B,1,d)
        rgb_token, rgb_w = self.rgb_weight(rgb_t1, rgb_t2)  # (B,1,d), (B,2)

        hs_t1 = self._encode_hs_one(hs_v1)         # (B,1,d)
        hs_t2 = self._encode_hs_one(hs_v2)         # (B,1,d)
        hs_token, hs_w = self.hs_weight(hs_t1, hs_t2)

        # 模态嵌入
        spec_mod = self.modality_emb(torch.zeros(1, dtype=torch.long, device=spectral_vec.device)).expand(B, spec_spad_tokens.shape[1], -1)
        rgb_mod  = self.modality_emb(torch.ones(1, dtype=torch.long, device=rgb_v1.device)).expand(B, 1, -1)
        hs_mod   = self.modality_emb(torch.tensor(2, device=hs_v1.device)).expand(B, 1, -1)

        spec_spad_tokens = spec_spad_tokens + spec_mod
        rgb_token        = rgb_token + rgb_mod
        hs_token         = hs_token + hs_mod

        cls = self.cls_token.expand(B, 1, -1)
        tokens = torch.cat([cls, spec_spad_tokens, rgb_token, hs_token], dim=1)
        attn_mask = torch.ones(B, tokens.shape[1], device=tokens.device)
        out = self.transformer(inputs_embeds=tokens, attention_mask=attn_mask).last_hidden_state
        cls_out = out[:, 0, :]
        pred = self.head(cls_out)  # (B,1)

        return pred, {"rgb_w": rgb_w, "hs_w": hs_w}
