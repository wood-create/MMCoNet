import torch
import torch.nn as nn
from torchvision import models

def _load_resnet101_backbone():
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    return nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
    )

def _load_resnet18_backbone():
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
        resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
    )

class PromptBlock(nn.Module):
    def __init__(self, prompt_len, d_model, num_tasks):
        super().__init__()
        self.prompt_len = prompt_len
        self.num_tasks = num_tasks
        self.prompt_table = nn.Parameter(torch.randn(num_tasks, prompt_len, d_model))

    def forward(self, x, task_id):
        B = x.size(0)
        p = self.prompt_table[task_id].unsqueeze(0).expand(B, -1, -1)
        return torch.cat([p, x], dim=1)

class CrossModalAdapter(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model)
        )

    def forward(self, q, kv):
        qn = self.norm(q)
        kn = self.norm(kv)
        a, _ = self.cross_attn(qn, kn, kn)
        return q + self.adapter(a)

class CoCaEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.encoder(x))

class ViLTCoCaModel_CROME_PromptMulti(nn.Module):
    def __init__(self, num_spec_feats, d_model=256, prompt_len=4, freeze_cnn=False):
        super().__init__()
        self.spec_proj = nn.Linear(num_spec_feats, d_model)
        self.spec_prompt = PromptBlock(prompt_len, d_model, num_tasks=2)
        self.img_prompt = PromptBlock(prompt_len, d_model, num_tasks=2)

        self.rgb_enc = nn.Sequential(
            _load_resnet101_backbone(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, d_model),
            nn.LayerNorm(d_model)
        )
        self.gray_enc = nn.Sequential(
            _load_resnet18_backbone(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model)
        )
        if freeze_cnn:
            for p in self.rgb_enc.parameters(): p.requires_grad = False
            for p in self.gray_enc.parameters(): p.requires_grad = False

        self.spec_transformer = CoCaEncoder(d_model=d_model)
        self.img_transformer = CoCaEncoder(d_model=d_model)
        self.cross_adapter_s2i = CrossModalAdapter(d_model=d_model)
        self.cross_adapter_i2s = CrossModalAdapter(d_model=d_model)

        self.contrast_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 128)
        )
        self.reg_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, spectral_vec, rgb_img, gray_img):
        B = spectral_vec.size(0)
        spec_token = self.spec_proj(spectral_vec).unsqueeze(1)
        rgb_feat = self.rgb_enc(rgb_img).unsqueeze(1)
        gray_feat = self.gray_enc(gray_img).unsqueeze(1)
        img_token = torch.cat([rgb_feat, gray_feat], dim=1)

        spec_tok = self.spec_prompt(spec_token, task_id=0)
        img_tok = self.img_prompt(img_token, task_id=1)

        spec_encoded = self.spec_transformer(spec_tok)
        img_encoded = self.img_transformer(img_tok)

        spec_fused = self.cross_adapter_s2i(spec_encoded, img_encoded)
        img_fused = self.cross_adapter_i2s(img_encoded, spec_encoded)

        spec_feat = spec_fused.mean(dim=1)
        img_feat = img_fused.mean(dim=1)

        spec_c = self.contrast_proj(spec_feat)
        img_c = self.contrast_proj(img_feat)
        spad = self.reg_head(torch.cat([spec_feat, img_feat], dim=1)).squeeze(1)
        return spad, spec_c, img_c
