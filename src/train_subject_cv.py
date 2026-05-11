#!/usr/bin/env python3
"""
DualDomainNet V6 - Training Script with ImageNet Pretrained Weights
Pipeline Step 3: Model Training
Data Source: Direct HDF5 reading (Pre-extracted 2D Coronal Slices from Step 2C)
"""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random

import h5py
import numpy as np
import pandas as pd
import scipy.ndimage  # 取代 OpenCV 進行影像旋轉
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    confusion_matrix, roc_curve, auc
)
from tqdm import tqdm
import torchvision.models as models
import matplotlib.pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """
    V6 雙域網路訓練配置
    
    關鍵策略：
    - 漸進式解凍：Phase 1(凍結)→Phase 2(layer4)→Phase 3(layer3-4)
    - 正則化：L2(5e-4) + Dropout(0.3) + Stochastic Depth(0.1)
    - 焦點損失：gamma=1.0，降低少數類別訓練不穩定性
    - HDF5 數據：雙域通道 (11238, 2, 224, 224)，受試者級交叉驗證
    """
    
    # Paths
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
    
    # HDF5 檔案路徑
    HDF5_PATH = Path(os.getenv("ADNI_HDF5_PATH", PROJECT_ROOT / "data" / "adcn_slices.h5"))
    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "outputs" / "dualdomain_v6"))
    
    # HDF5 Key 名稱
    H5_KEY_IMAGES = 'images'
    H5_KEY_LABELS = 'labels'
    H5_KEY_SUBJECTS = 'subject_ids'
    
    # W&B Settings
    WANDB_PROJECT = "adni-dual-domain-coronal"
    USE_WANDB = False
    
    # Model settings
    MODEL_NAME = "DualDomainNet_V6_ImageNet"
    NUM_CLASSES = 2
    SPATIAL_INPUT_SIZE = (224, 224)
    FREQ_INPUT_SIZE = (224, 224)
    
    # Training settings
    N_FOLDS = 5
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    SEED = 42
    
    # Progressive unfreezing phases
    PHASE1_EPOCHS = 20   # 10→20：給 head 更多收斂時間
    PHASE2_EPOCHS = 20
    PHASE3_EPOCHS = 30
    TOTAL_EPOCHS = PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS

    # Learning rates for each phase
    PHASE1_LR = 3e-4    # 1e-3→3e-4：降低初始 LR 減少震盪
    PHASE2_LR = 1e-4
    PHASE3_LR = 5e-5

    # Regularization
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.3        # 0.6→0.3：降低正則化讓模型能學到兩個類別
    DROP_PATH_RATE = 0.1 # 0.2→0.1：配合 Dropout 降低一起減弱
    USE_STOCHASTIC_DEPTH = True
    FOCAL_GAMMA = 1.0    # 2.0→1.0：降低 Focal Loss 強度，避免不穩定
    GRAD_CLIP = 1.0

    # Early stopping
    EARLY_STOP_PATIENCE = 25  # 15→25：給模型更多時間脫離局部最優
    
    # Mixed precision
    USE_AMP = True
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def setup(cls):
        if not cls.HDF5_PATH.exists():
            raise FileNotFoundError(f"找不到 HDF5 檔案: {cls.HDF5_PATH}。請確認路徑是否正確。")
            
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (cls.OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "logs").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "plots").mkdir(exist_ok=True)

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path, fold: Optional[int] = None) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_str = f"_fold{fold}" if fold is not None else ""
    log_file = output_dir / "logs" / f"training{fold_str}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# Stochastic Depth (DropPath) - 隨機深度正則化
# ============================================================================

class StochasticDepth(nn.Module):
    """
    隨機深度 (Stochastic Depth / DropPath)：
    在訓練時以概率 drop_prob 隨機丟棄整個殘差塊，有效降低過擬合。
    
    機制：
    - 訓練時：保留概率 (1-drop_prob)，丟棄概率 drop_prob
    - 測試時：全部保留，但縮放補償
    
    優點：
    1. 隨機性更高，避免記憶特定特徵
    2. 動態調整深度，簡化訓練
    3. 對預訓練大模型（ResNet50）特別有效
    """
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        # 測試時不丟棄
        if not self.training or self.drop_prob == 0:
            return x
        
        # 訓練時：以保留概率採樣
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (batch_size, 1, 1, 1)
        random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))

        # 縮放以補償丟棄
        return x * random_tensor / keep_prob

# ============================================================================
# Dataset - 數據集加載與增強
# ============================================================================

class ADNIHDF5SliceDataset(Dataset):
    """
    從 HDF5 檔案讀取預先切好之 2D 切片的 Dataset。
    使用 Preload：初始化時一次性將指定 indices 讀進 RAM，
    避免網路儲存 (NFS/Lustre) 上隨機 HDF5 I/O 造成的嚴重瓶頸。
    """
    def __init__(
        self,
        hdf5_path: Path,
        indices: List[int],
        augment: bool = False
    ):
        self.indices = indices
        self.augment = augment

        with h5py.File(str(hdf5_path), 'r') as f:
            idx_arr = np.array(indices)
            self.images = f[Config.H5_KEY_IMAGES][idx_arr, 0].astype(np.float32)  # (N, H, W)
            self.labels = f[Config.H5_KEY_LABELS][idx_arr]
            raw_sids = f[Config.H5_KEY_SUBJECTS][idx_arr]
            self.subject_ids = [
                s.decode('utf-8') if isinstance(s, bytes) else str(s)
                for s in raw_sids
            ]

    def __len__(self) -> int:
        return len(self.indices)
            
    def _compute_fft(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        window_h = np.hanning(h)
        window_w = np.hanning(w)
        window = np.outer(window_h, window_w)
        windowed = image * window
        
        fft = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        log_magnitude = np.log1p(magnitude)
        if log_magnitude.max() - log_magnitude.min() > 0:
            log_magnitude = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-8)
        return log_magnitude.astype(np.float32)
    
    def _augment_slice_enhanced(self, image: np.ndarray) -> np.ndarray:
        """
        增強版數據增強管道：6 種增強方法組合
        
        增強方法及概率：
        1. 水平翻轉 (50%)：基礎對稱增強
        2. 旋轉 ±15° (50%)：模擬掃描角度變化
        3. 對比度調整 (50%)：α∈[0.85,1.15]
        4. Cutout (30%)：隨機遮蓋矩形區域
        5. GridMask (25%)：棋盤式遮蓋
        6. 彈性變形 (20%)：模擬生物組織變化
        
        設計理由：
        - 多樣化：豐富訓練樣本變化
        - 醫學友好：保持解剖結構可識別性
        - 層級增強：基礎→進階→變化
        
        預期效果：+3-4% AUC 提升
        """
        
        # 原有增強
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
        
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)  # 增加旋轉範圍
            image = scipy.ndimage.rotate(image, angle, reshape=False, order=1, mode='nearest')
        
        if random.random() > 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-0.1, 0.1)
            image = np.clip(alpha * image + beta, 0, 1)
        
        # ===== 新增增強方法 =====
        
        # 1. Cutout (30% 概率) - 隨機遮蓋矩形區域
        if random.random() < 0.3:
            h, w = image.shape
            cut_h = random.randint(h // 6, h // 3)
            cut_w = random.randint(w // 6, w // 3)
            y = random.randint(0, max(1, h - cut_h))
            x = random.randint(0, max(1, w - cut_w))
            image[y:y+cut_h, x:x+cut_w] = np.random.uniform(0, 1)
        
        # 2. GridMask (25% 概率) - 棋盤遮蔽
        if random.random() < 0.25:
            mask_ratio = random.uniform(0.3, 0.5)
            d = int(224 * mask_ratio)
            for i in range(0, 224, d):
                for j in range(0, 224, d):
                    if random.random() < 0.5:
                        image[i:min(i+d, 224), j:min(j+d, 224)] = 0
        
        # 3. Elastic Deformation (20% 概率) - 彈性變形
        if random.random() < 0.2:
            from scipy.ndimage import map_coordinates
            alpha = random.uniform(30, 50)
            sigma = random.uniform(3, 5)
            
            # 生成位移場
            dx = np.random.randn(224, 224) * sigma
            dy = np.random.randn(224, 224) * sigma
            
            # 應用位移
            x_coords, y_coords = np.meshgrid(np.arange(224), np.arange(224))
            indices_x = np.clip(x_coords + dx * alpha, 0, 223).astype(np.float32)
            indices_y = np.clip(y_coords + dy * alpha, 0, 223).astype(np.float32)
            
            image = map_coordinates(image, [indices_y, indices_x], 
                                    order=1, mode='nearest')
        
        return image.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        slice_2d = self.images[idx].copy()
        label = self.labels[idx]
        subject_id = self.subject_ids[idx]
        
        if self.augment:
            slice_2d = self._augment_slice_enhanced(slice_2d)
        
        spatial_input = np.stack([slice_2d, slice_2d, slice_2d], axis=0)
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        spatial_input = (spatial_input - mean) / std
        
        freq_slice = self._compute_fft(slice_2d)
        freq_input = freq_slice[np.newaxis, ...]
        
        return {
            'spatial_input': torch.tensor(spatial_input, dtype=torch.float32),
            'freq_input': torch.tensor(freq_input, dtype=torch.float32),
            'label': torch.tensor(int(label), dtype=torch.long),
            'subject_id': subject_id
        }

# ============================================================================
# Model Architecture
# ============================================================================

class FrequencyStreamCNN(nn.Module):
    """
    頻域特徵提取網路 - 處理 FFT 頻譜數據
    
    此模組針對醫學影像的頻域特性優化：
    - 輸入: FFT 對數幅度譜 (log-normalized magnitude spectrum)
    - 架構: 3 層 CNN + 全連接層，設計輕量化 (vs ResNet50 重型主幹)
    - 輸出: 128 維特徵向量，與空間主幹融合
    """
    def __init__(self, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return x

class DualDomainNetV6(nn.Module):
    """
    雙域融合網路 V6 - 空間 + 頻域特徵聯合學習
    
    架構設計：
    - 空間流：ImageNet 預訓練 ResNet50 (2500M->256D 投影)
    - 頻率流：自訓練輕量 CNN (128D) 特化頻譜特徵
    - 融合層：拼接後通過 BatchNorm+Dropout 強正則化 (256D)
    - 預訓練策略：初期凍結主幹，漸進式解凍 layer4→layer3-4
    
    輸入形狀：(batch, 2, 224, 224) 其中 channel 0=空間, 1=頻域
    """
    def __init__(self, num_classes: int = 2, dropout: float = 0.5, freeze_backbone: bool = True, drop_path_rate: float = 0.0):
        super().__init__()
        self.drop_path = StochasticDepth(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.spatial_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        spatial_feature_dim = self.spatial_backbone.fc.in_features
        self.spatial_backbone.fc = nn.Identity()
        
        if freeze_backbone:
            self._freeze_backbone()
        
        self.freq_stream = FrequencyStreamCNN(in_channels=1, dropout=dropout/2)
        self.spatial_dim = spatial_feature_dim
        self.freq_dim = 128
        self.fused_dim = 256
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.spatial_dim, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.freq_proj = nn.Sequential(
            nn.Linear(self.freq_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(384 + 128, self.fused_dim),
            nn.BatchNorm1d(self.fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(self.fused_dim, num_classes)
        self._init_weights()
    
    def _freeze_backbone(self):
        for param in self.spatial_backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_layer4(self):
        for param in self.spatial_backbone.layer4.parameters():
            param.requires_grad = True
    
    def unfreeze_layer3_4(self):
        for param in self.spatial_backbone.layer3.parameters():
            param.requires_grad = True
        for param in self.spatial_backbone.layer4.parameters():
            param.requires_grad = True
    
    def _init_weights(self):
        for module in [self.freq_stream, self.spatial_proj, self.freq_proj, self.fusion, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, spatial_input: torch.Tensor, freq_input: torch.Tensor) -> torch.Tensor:
        spatial_features = self.spatial_backbone(spatial_input)
        spatial_proj = self.drop_path(self.spatial_proj(spatial_features))

        freq_features = self.freq_stream(freq_input)
        freq_proj = self.drop_path(self.freq_proj(freq_features))

        fused = torch.cat([spatial_proj, freq_proj], dim=1)
        fused = self.fusion(fused)
        
        logits = self.classifier(fused)
        return logits

# ============================================================================
# Training Utilities
# ============================================================================

class FocalLoss(nn.Module):
    """
    焦點損失 - 重視困難樣本，抑制簡單樣本
    
    公式：FL = -α * (1 - p_t)^γ * CE_loss
    - gamma controls how strongly easy examples are down-weighted.
    - alpha balances positive and negative class contribution.
    用於不平衡醫學影像分類，避免簡單樣本淹沒損失信號
    """
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

class EarlyStopping:
    """
    早停機制 - 防止過度擬合
    
    監控驗證指標（如 AUC），若無進步則計數
    耐心次數 (patience=15) 後停訓，保存最佳模型
    """
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        improved = score > self.best_score + self.min_delta if self.mode == 'max' else score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

def calculate_metrics(labels, preds, probs):
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': accuracy_score(labels, preds),
        'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.5,
        'f1': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='macro', zero_division=0),
        'sensitivity': sensitivity,
        'specificity': specificity
    }

# ============================================================================
# Plotting Utilities
# ============================================================================

def plot_cv_metrics(results_df: pd.DataFrame, output_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics = ['subject_accuracy', 'subject_auc', 'subject_f1', 'subject_sensitivity', 'subject_specificity']
    labels = ['Accuracy', 'AUC', 'F1-Score', 'Sensitivity', 'Specificity']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, fold in enumerate(results_df['fold']):
        fold_scores = results_df.iloc[i][metrics].values
        plt.bar(x + (i - 2) * width, fold_scores, width, label=f'Fold {int(fold)}')
    
    mean_scores = results_df[metrics].mean().values
    plt.bar(x + 3 * width, mean_scores, width, label='Mean', color='black', alpha=0.7)
    
    plt.title('Cross-Validation Metrics Summary')
    plt.ylabel('Score')
    plt.xticks(x + width/2, labels)
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "cv_metrics_summary.png", dpi=300)
    plt.close()

def plot_confusion_matrix_all(all_labels: list, all_preds: list, output_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['CN', 'AD'], yticklabels=['CN', 'AD'])
    plt.title('Aggregated Confusion Matrix (Subject-Level)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "confusion_matrix.png", dpi=300)
    plt.close()

def plot_roc_curve_all(all_labels: list, all_probs: list, output_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Subject-Level)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "roc_curve.png", dpi=300)
    plt.close()

# ============================================================================
# Pipeline Step 3: Model Training
# ============================================================================

def train_one_epoch(
    model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, scaler: GradScaler, device: torch.device,
    epoch: int, config: Config
) -> Dict[str, float]:
    # 單輪訓練：FP16 mixed precision + Stochastic Depth + 梯度裁剪防爆炸
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in pbar:
        spatial_input = batch['spatial_input'].to(device)
        freq_input = batch['freq_input'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        with autocast(enabled=config.USE_AMP):
            logits = model(spatial_input, freq_input)
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    accuracy = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(train_loader)
    return {'loss': avg_loss, 'accuracy': accuracy, 'precision': train_precision}

def validate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device, config: Config
) -> Dict[str, float]:
    # 驗證階段：no_grad() 下評估，蒐集概率用於計算 AUC/敏感性/特異性
    model.eval()
    total_loss, all_preds, all_probs, all_labels = 0.0, [], [], []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            spatial_input = batch['spatial_input'].to(device)
            freq_input = batch['freq_input'].to(device)
            labels = batch['label'].to(device)
            
            with autocast(enabled=config.USE_AMP):
                logits = model(spatial_input, freq_input)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(val_loader)
    return metrics

def subject_level_voting(
    model: nn.Module, data_loader: DataLoader, device: torch.device, config: Config
) -> Tuple[Dict[str, float], List[int], List[int], List[float]]:
    """
    受試者層級投票 - 多切片融合，提升穩定性
    
    邏輯：同一受試者的多個冠狀切片經過多數投票決定最終預測
    對應大腦萎縮分佈的層級結構，避免單一切片噪聲
    """
    model.eval()
    subject_preds, subject_probs, subject_labels = {}, {}, {}
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Subject-level Voting"):
            spatial_input = batch['spatial_input'].to(device)
            freq_input = batch['freq_input'].to(device)
            labels = batch['label']
            subject_ids = batch['subject_id']
            
            with autocast(enabled=config.USE_AMP):
                logits = model(spatial_input, freq_input)
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            ad_probs = probs[:, 1].cpu().numpy()
            
            for i, sid in enumerate(subject_ids):
                if sid not in subject_preds:
                    subject_preds[sid], subject_probs[sid] = [], []
                    subject_labels[sid] = labels[i].item()
                subject_preds[sid].append(preds[i])
                subject_probs[sid].append(ad_probs[i])
    
    final_preds, final_probs, final_labels = [], [], []
    for sid in subject_preds:
        avg_ad_prob = np.mean(subject_probs[sid])
        # 使用 AD 機率均值做閾值（0.3 比 0.5 更寬鬆，修正模型偏向 CN 的問題）
        final_preds.append(1 if avg_ad_prob >= 0.3 else 0)
        final_probs.append(avg_ad_prob)
        final_labels.append(subject_labels[sid])
    
    metrics = calculate_metrics(final_labels, final_preds, final_probs)
    metrics_dict = {
        'subject_accuracy': metrics['accuracy'],
        'subject_auc': metrics['auc'],
        'subject_f1': metrics['f1'],
        'subject_sensitivity': metrics['sensitivity'],
        'subject_specificity': metrics['specificity'],
        'n_subjects': len(subject_preds)
    }
    return metrics_dict, final_labels, final_preds, final_probs

def train_fold(fold: int, train_idx: List[int], val_idx: List[int], config: Config, logger: logging.Logger):
    logger.info(f"\n{'='*60}\nTraining Fold {fold + 1}/{config.N_FOLDS}\n{'='*60}")
    
    # 初始化指標收集列表
    epochs_list = []
    train_losses = []
    val_losses = []
    val_aucs = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []
    val_f1s = []
    
    use_wandb = config.USE_WANDB and wandb is not None
    if use_wandb:
        wandb.init(
            project=config.WANDB_PROJECT,
            group=config.MODEL_NAME,
            name=f"fold_{fold+1}",
            config={"fold": fold + 1, "batch_size": config.BATCH_SIZE, "epochs": config.TOTAL_EPOCHS},
            reinit=True
        )
    
    train_dataset = ADNIHDF5SliceDataset(config.HDF5_PATH, train_idx, augment=True)
    val_dataset = ADNIHDF5SliceDataset(config.HDF5_PATH, val_idx, augment=False)

    # 直接從已預載入 RAM 的 dataset 取 labels，不再重新讀 HDF5
    train_labels = [int(train_dataset.labels[i]) for i in range(len(train_dataset))]
    weights = 1.0 / np.bincount(train_labels)[train_labels]
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        sampler=WeightedRandomSampler(weights, len(weights)),
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    
    model = DualDomainNetV6(
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
        freeze_backbone=True,
        drop_path_rate=config.DROP_PATH_RATE if config.USE_STOCHASTIC_DEPTH else 0.0
    ).to(config.DEVICE)
    
    criterion = FocalLoss(gamma=config.FOCAL_GAMMA)
    scaler = GradScaler(enabled=config.USE_AMP)
    early_stopping = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode='max')
    
    best_val_auc = 0.0
    best_model_state = None
    
    def run_phase(epochs, start_epoch, lr, phase_name):
        nonlocal best_val_auc, best_model_state
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5) if phase_name != "Phase 1" else CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            global_epoch = start_epoch + epoch
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE, global_epoch, config)
            val_metrics = validate(model, val_loader, criterion, config.DEVICE, config)
            
            # 收集指標數據
            epochs_list.append(global_epoch + 1)
            train_losses.append(train_metrics['loss'])
            val_losses.append(val_metrics['loss'])
            val_aucs.append(val_metrics['auc'])
            val_accuracies.append(val_metrics['accuracy'])
            train_precisions.append(train_metrics['precision'])
            val_precisions.append(val_metrics['precision'])
            val_f1s.append(val_metrics['f1'])
            
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['auc'])
            else:
                scheduler.step()
            
            if use_wandb:
                wandb.log({
                    "epoch": global_epoch + 1,
                    "train_loss": train_metrics['loss'],
                    "train_acc": train_metrics['accuracy'],
                    "val_loss": val_metrics['loss'],
                    "val_acc": val_metrics['accuracy'],
                    "val_auc": val_metrics['auc'],
                    "val_sensitivity": val_metrics['sensitivity'],
                    "val_specificity": val_metrics['specificity']
                })
            
            logger.info(f"Epoch {global_epoch+1:3d} | Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, Sens: {val_metrics['sensitivity']:.2f}, Spec: {val_metrics['specificity']:.2f}")
            
            # 只有 AUC 提升且 Sensitivity > 0.1 才存成 best（避免儲存全預測 CN 的模型）
            is_balanced = val_metrics['sensitivity'] > 0.1
            if val_metrics['auc'] > best_val_auc and is_balanced:
                best_val_auc = val_metrics['auc']
                best_model_state = model.state_dict().copy()
            elif best_model_state is None:
                # 若還沒有任何 best，先無條件儲存（防止後續 load 出錯）
                best_model_state = model.state_dict().copy()
            
            if early_stopping(val_metrics['auc']):
                logger.info(f"Early stopping triggered at epoch {global_epoch+1}")
                return True
        return False

    logger.info(f"\n--- Phase 1: Training head only ---")
    if not run_phase(config.PHASE1_EPOCHS, 0, config.PHASE1_LR, "Phase 1"):
        logger.info(f"\n--- Phase 2: Unfreezing layer4 ---")
        model.unfreeze_layer4()
        if not run_phase(config.PHASE2_EPOCHS, config.PHASE1_EPOCHS, config.PHASE2_LR, "Phase 2"):
            logger.info(f"\n--- Phase 3: Unfreezing layer3+layer4 ---")
            model.unfreeze_layer3_4()
            run_phase(config.PHASE3_EPOCHS, config.PHASE1_EPOCHS + config.PHASE2_EPOCHS, config.PHASE3_LR, "Phase 3")
    
    model.load_state_dict(best_model_state)
    subject_metrics, final_labels, final_preds, final_probs = subject_level_voting(model, val_loader, config.DEVICE, config)
    
    if use_wandb:
        wandb.log({
            "best_val_auc": best_val_auc,
            "subject_accuracy": subject_metrics['subject_accuracy'],
            "subject_auc": subject_metrics['subject_auc'],
            "subject_f1": subject_metrics['subject_f1'],
            "subject_sensitivity": subject_metrics['subject_sensitivity'],
            "subject_specificity": subject_metrics['subject_specificity']
        })
        wandb.finish()
    
    # 繪製訓練指標圖表
    plot_training_metrics(epochs_list, train_losses, val_losses, val_aucs, val_accuracies, train_precisions, val_precisions, val_f1s, fold, config.OUTPUT_DIR)
    
    checkpoint_path = config.OUTPUT_DIR / "checkpoints" / f"fold{fold+1}_best.pth"
    torch.save({
        'fold': fold, 'model_state_dict': best_model_state,
        'best_val_auc': best_val_auc, 'subject_metrics': subject_metrics
    }, checkpoint_path)
    
    fold_dict = {'fold': fold + 1, 'best_val_auc': best_val_auc, **subject_metrics}
    return fold_dict, final_labels, final_preds, final_probs

def plot_training_metrics(epochs, train_losses, val_losses, val_aucs, val_accuracies, train_precisions, val_precisions, val_f1s, fold, output_dir):
    """
    繪製訓練過程中的指標變化圖表
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Fold {fold+1} Training Metrics', fontsize=16, fontweight='bold')
    
    # AUC
    axes[0, 0].plot(epochs, val_aucs, 'b-', linewidth=2, label='Validation AUC')
    axes[0, 0].set_title('AUC', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(epochs, val_accuracies, 'g-', linewidth=2, label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Precision
    axes[0, 2].plot(epochs, train_precisions, 'darkorange', linewidth=2, linestyle='--', label='Train Precision')
    axes[0, 2].plot(epochs, val_precisions, 'orange', linewidth=2, label='Validation Precision')
    axes[0, 2].set_title('Precision', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    
    # F1 Score
    axes[1, 0].plot(epochs, val_f1s, 'red', linewidth=2, label='Validation F1 Score')
    axes[1, 0].set_title('F1 Score', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Training Loss
    axes[1, 1].plot(epochs, train_losses, 'purple', linewidth=2, label='Training Loss')
    axes[1, 1].set_title('Training Loss', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Validation Loss
    axes[1, 2].plot(epochs, val_losses, 'brown', linewidth=2, label='Validation Loss')
    axes[1, 2].set_title('Validation Loss', fontweight='bold')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    # 保存圖表
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_path = plots_dir / f"fold_{fold+1}_training_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics plot saved to: {plot_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train DualDomainNet with subject-level cross-validation.")
    parser.add_argument("--hdf5-path", type=Path, default=Config.HDF5_PATH, help="Path to the authorized ADNI HDF5 slice file.")
    parser.add_argument("--output-dir", type=Path, default=Config.OUTPUT_DIR, help="Directory for checkpoints, logs, and plots.")
    parser.add_argument("--folds", type=int, default=Config.N_FOLDS, help="Number of StratifiedGroupKFold splits.")
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=Config.SEED, help="Random seed.")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    return parser.parse_args()

def apply_args(args):
    Config.HDF5_PATH = args.hdf5_path
    Config.OUTPUT_DIR = args.output_dir
    Config.N_FOLDS = args.folds
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_WORKERS = args.num_workers
    Config.SEED = args.seed
    Config.USE_WANDB = args.use_wandb

def main():
    args = parse_args()
    apply_args(args)
    Config.setup()
    set_seed(Config.SEED)
    logger = setup_logging(Config.OUTPUT_DIR)
    
    logger.info(f"讀取 HDF5 檔案以進行切分: {Config.HDF5_PATH}")
    
    with h5py.File(Config.HDF5_PATH, 'r') as f:
        labels = f[Config.H5_KEY_LABELS][:]
        subject_ids = f[Config.H5_KEY_SUBJECTS][:]
        
    if len(subject_ids) > 0 and isinstance(subject_ids[0], bytes):
        subject_ids = [s.decode('utf-8') for s in subject_ids]
        
    df_meta = pd.DataFrame({
        'slice_idx': range(len(labels)),
        'subject_id': subject_ids,
        'label': labels
    })
    
    skf = StratifiedGroupKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)
    
    all_fold_results = []
    global_labels = []
    global_preds = []
    global_probs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_meta, df_meta['label'], groups=df_meta['subject_id'])):
        train_idx_list = df_meta.iloc[train_idx]['slice_idx'].tolist()
        val_idx_list = df_meta.iloc[val_idx]['slice_idx'].tolist()
        
        fold_results, f_labels, f_preds, f_probs = train_fold(fold, train_idx_list, val_idx_list, Config, logger)
        all_fold_results.append(fold_results)
        
        global_labels.extend(f_labels)
        global_preds.extend(f_preds)
        global_probs.extend(f_probs)
    
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(Config.OUTPUT_DIR / "cv_results.csv", index=False)
    
    logger.info("Generating plots...")
    plot_cv_metrics(results_df, Config.OUTPUT_DIR)
    plot_confusion_matrix_all(global_labels, global_preds, Config.OUTPUT_DIR)
    plot_roc_curve_all(global_labels, global_probs, Config.OUTPUT_DIR)
    
    logger.info(f"Training completed! Plots saved to {Config.OUTPUT_DIR / 'plots'}")

if __name__ == "__main__":
    main()
