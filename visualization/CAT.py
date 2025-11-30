"""
Class activation topography (CAT) for EEG Conformer model visualization
結合 Class Activation Map (CAM) 和 EEG 拓撲圖來可視化模型關注的腦區

適配 EEG Conformer 模型：
- 使用 Conformer 模型架構
- 支援 19 通道 EEG 數據
- 3 類別分類（AD/FTD/CN）

refer to high-star repo on github: 
https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/grad_cam

Salute every open-source researcher and developer!
"""

import argparse
import os
gpus = [0]  # 修改為您的 GPU 編號
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange
import matplotlib.pyplot as plt
from torch.backends import cudnn
import mne

# 導入 Conformer 模型
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conformer import Conformer

# 導入 GradCAM 工具
from utils import GradCAM

cudnn.benchmark = False
cudnn.deterministic = True

# ==================== 配置參數 ====================
# 模型參數
MODEL_PATH = './checkpoints/best_model.pth'  # 模型路徑
EMB_SIZE = 40
DEPTH = 6
N_CLASSES = 3  # AD/FTD/CN

# 數據參數
DATA_PATH = None  # 如果為 None，將從評估腳本載入數據
# 或者直接指定數據文件路徑，例如：
# DATA_PATH = './results/test_data.npy'  # 從評估結果載入
# 或者從原始數據集載入（需要實現數據載入函數）

# 可視化參數
TARGET_CATEGORY = 2  # 要可視化的類別：0=AD, 1=FTD, 2=CN
N_SAMPLES = None  # 要處理的樣本數，如果為 None 則處理所有樣本
OUTPUT_DIR = './results/visualization'  # 輸出目錄

# 設備設定
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"使用設備: {device}")

# ==================== 數據載入函數 ====================
def load_test_data_from_evaluator():
    """
    從評估腳本載入測試數據
    這需要先運行評估腳本生成數據
    """
    try:
        # 嘗試從評估結果載入（多個可能的路徑）
        possible_data_paths = [
            './results/test_data.npy',
            './results/val_data.npy',  # 如果只有驗證集數據
        ]
        possible_label_paths = [
            './results/test_labels.npy',
            './results/val_labels.npy',  # 如果只有驗證集標籤
        ]
        
        data_path = None
        label_path = None
        
        # 查找存在的數據文件
        for path in possible_data_paths:
            if os.path.exists(path):
                data_path = path
                print(f"找到數據文件: {data_path}")
                break
        
        # 查找存在的標籤文件
        for path in possible_label_paths:
            if os.path.exists(path):
                label_path = path
                print(f"找到標籤文件: {label_path}")
                break
        
        if data_path and label_path:
            data = np.load(data_path)
            labels = np.load(label_path)
            print(f"從評估結果載入數據: {data.shape}, 標籤: {labels.shape}")
            return data, labels
        elif data_path:
            # 只有數據文件，沒有標籤文件
            data = np.load(data_path)
            print(f"從評估結果載入數據: {data.shape} (無標籤文件)")
            return data, None
        else:
            print("警告: 找不到評估結果數據文件")
            print("  請先運行 evaluate_conformer_model.py 生成數據")
            print("  或者手動指定 DATA_PATH 參數")
            return None, None
    except Exception as e:
        print(f"載入數據時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_data_from_dataset(base_dir='/ibmnas/427/bachelors/b12901077/eeg', 
                           dataset_dir=None, 
                           n_samples=None):
    """
    直接從數據集載入測試數據
    """
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    
    if dataset_dir is None:
        dataset_dir = os.path.join(base_dir, 'ds004504')
    
    # 這裡簡化處理，實際應該使用與訓練時相同的數據載入邏輯
    # 建議先運行評估腳本生成數據文件
    print("請使用 load_test_data_from_evaluator() 或先運行評估腳本")
    return None, None

# ==================== 模型包裝器 ====================
class ConformerWrapper(nn.Module):
    """
    包裝 Conformer 模型，使其只返回分類輸出（而不是 (features, output) tuple）
    這樣可以適配 GradCAM 的要求
    """
    def __init__(self, conformer_model):
        super().__init__()
        self.conformer = conformer_model
    
    def forward(self, x):
        # Conformer 返回 (features, output)，我們只需要 output
        _, output = self.conformer(x)
        return output

# ==================== 模型載入 ====================
def load_model(model_path, emb_size=40, depth=6, n_classes=3):
    """載入訓練好的 Conformer 模型並包裝"""
    model = Conformer(emb_size=emb_size, depth=depth, n_classes=n_classes)
    
    # 載入模型權重
    if os.path.exists(model_path):
        print(f"載入模型: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # 處理 DataParallel 包裝的模型
        if any(key.startswith('module.') for key in state_dict.keys()):
            # 如果模型是用 DataParallel 保存的，需要移除 'module.' 前綴
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # 用包裝器包裝模型，使其只返回分類輸出
        wrapped_model = ConformerWrapper(model)
        wrapped_model = wrapped_model.to(device)
        wrapped_model.eval()
        
        print("模型載入成功")
        return wrapped_model
    else:
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

# ==================== Reshape Transform ====================
def reshape_transform(tensor):
    """
    適配 Conformer Transformer 輸出的 reshape 函數
    Conformer 的 Transformer 輸出格式: (batch, n_patches, emb_size)
    需要轉換為: (batch, emb_size, 1, n_patches) 以適配 GradCAM
    """
    # tensor shape: (b, n, e) -> (b, e, 1, n)
    # 使用 unsqueeze 插入維度，然後 permute 重新排列
    # 或者直接使用正確的 rearrange 模式
    result = rearrange(tensor, 'b n e -> b e 1 n')
    return result

# ==================== EEG 通道映射 ====================
def create_eeg_info(n_channels=19, sfreq=500.0):
    """
    創建 19 通道 EEG 的 MNE info 對象
    使用標準 10-20 系統的通道名稱
    """
    # 標準 19 通道 10-20 系統通道名稱
    standard_19_channels = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
    ]
    
    # 如果通道數不是 19，使用前 n_channels 個
    if n_channels <= len(standard_19_channels):
        ch_names = standard_19_channels[:n_channels]
    else:
        ch_names = standard_19_channels + [f'EEG{i}' for i in range(len(standard_19_channels), n_channels)]
    
    # 創建標準 10-20 montage
    montage = mne.channels.make_standard_montage('standard_1020')
    
    # 選擇對應的通道（只保留在 montage 中存在的通道）
    available_ch_names = [ch for ch in ch_names if ch in montage.ch_names]
    
    if len(available_ch_names) < n_channels:
        print(f"警告: 只有 {len(available_ch_names)} 個通道在標準 montage 中找到，使用前 {n_channels} 個可用通道")
        # 使用標準 10-20 系統的前 n_channels 個可用通道
        if len(available_ch_names) >= n_channels:
            ch_names = available_ch_names[:n_channels]
        else:
            # 如果可用通道不足，使用所有可用通道並補充虛擬通道名
            ch_names = available_ch_names
            while len(ch_names) < n_channels:
                ch_names.append(f'EEG{len(ch_names)}')
    
    # 創建 info 對象（直接使用選定的通道名）
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # 設置 montage（MNE 會自動匹配可用的通道）
    try:
        info.set_montage(montage)
    except Exception as e:
        print(f"警告: 設置 montage 時出現問題: {e}")
        print("將使用默認的標準 10-20 montage")
        try:
            # 嘗試使用標準 montage
            standard_montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(standard_montage)
        except:
            print("無法設置 montage，將繼續使用默認設置")
    
    return info

# ==================== 主函數 ====================
def main():
    """主函數：執行 CAT 可視化"""
    
    # 創建輸出目錄
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"創建輸出目錄: {OUTPUT_DIR}")
    
    # 載入數據
    print("="*60)
    print("步驟 1: 載入數據")
    print("="*60)
    
    if DATA_PATH is not None and os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH)
        print(f"從指定路徑載入數據: {data.shape}")
        labels = None  # 如果數據文件不包含標籤
    else:
        data, labels = load_test_data_from_evaluator()
    
    if data is None:
        print("錯誤: 無法載入數據，請先運行評估腳本或指定數據路徑")
        return
    
    # 確保數據格式正確: (trials, 1, channels, timepoints) 或 (trials, channels, timepoints)
    if len(data.shape) == 3:
        # (trials, channels, timepoints) -> (trials, 1, channels, timepoints)
        data = np.expand_dims(data, axis=1)
    elif len(data.shape) == 4:
        # 已經是 (trials, 1, channels, timepoints)
        pass
    else:
        raise ValueError(f"不支持的數據形狀: {data.shape}")
    
    print(f"數據形狀: {data.shape}")
    
    # 如果指定了樣本數，只處理前 N 個樣本
    if N_SAMPLES is not None and N_SAMPLES < len(data):
        data = data[:N_SAMPLES]
        if labels is not None:
            labels = labels[:N_SAMPLES]
        print(f"只處理前 {N_SAMPLES} 個樣本")
    
    # 載入模型
    print("="*60)
    print("步驟 2: 載入模型")
    print("="*60)
    wrapped_model = load_model(MODEL_PATH, EMB_SIZE, DEPTH, N_CLASSES)
    
    # 設置目標層（Transformer Encoder 的最後一層）
    # Conformer 結構: [PatchEmbedding, TransformerEncoder, ClassificationHead]
    # TransformerEncoder 包含多個 TransformerEncoderBlock
    # 注意：wrapped_model.conformer 是原始的 Conformer 模型
    target_layers = [wrapped_model.conformer[1][-1]]  # Transformer Encoder 的最後一層
    print(f"目標層: {target_layers[0]}")
    
    # 創建 GradCAM
    print("="*60)
    print("步驟 3: 初始化 GradCAM")
    print("="*60)
    # 注意：GradCAM 需要使用包裝後的模型（只返回分類輸出）
    # 但目標層需要從原始模型中獲取
    cam = GradCAM(
        model=wrapped_model, 
        target_layers=target_layers, 
        use_cuda=USE_CUDA,
        reshape_transform=reshape_transform
    )
    
    # 計算 CAM
    print("="*60)
    print("步驟 4: 計算 Class Activation Map")
    print("="*60)
    print(f"處理 {len(data)} 個樣本...")
    
    all_cam = []
    for i in range(len(data)):
        if (i + 1) % 50 == 0:
            print(f"  處理進度: {i+1}/{len(data)}")
        
        test_input = torch.as_tensor(data[i:i+1, :, :, :], dtype=torch.float32)
        test_input = Variable(test_input, requires_grad=True)
        test_input = test_input.to(device)
        
        # 計算 CAM
        grayscale_cam = cam(input_tensor=test_input, target_category=TARGET_CATEGORY)
        grayscale_cam = grayscale_cam[0, :]  # 移除 batch 維度
        all_cam.append(grayscale_cam)
    
    print("CAM 計算完成")
    
    # 轉換為 numpy 數組
    all_cam = np.array(all_cam)  # (n_samples, n_patches, timepoints)
    print(f"CAM 形狀: {all_cam.shape}")
    
    # ==================== 計算平均值和可視化 ====================
    print("="*60)
    print("步驟 5: 生成可視化")
    print("="*60)
    
    # 計算所有數據的平均值
    # data shape: (n_samples, 1, 19, 1000)
    test_all_data = np.squeeze(np.mean(data, axis=0))  # (19, 1000)
    mean_all_test = np.mean(test_all_data, axis=1)  # (19,) - 每個通道的平均值
    
    # 計算所有 CAM 的平均值
    # all_cam shape: (n_samples, n_patches, timepoints)
    # 需要將 CAM 映射回通道維度
    # 這裡簡化處理：對時間維度求平均，然後映射到通道
    test_all_cam = np.mean(all_cam, axis=0)  # (n_patches, timepoints)
    mean_all_cam = np.mean(test_all_cam, axis=1)  # (n_patches,)
    
    # 將 CAM 的 patch 維度映射回通道維度
    # 由於 Conformer 的 patch embedding 會改變維度，這裡需要根據實際模型結構調整
    # 簡化處理：如果 CAM 的 patch 數與通道數不同，需要插值或平均
    if len(mean_all_cam) != len(mean_all_test):
        # 如果維度不匹配，使用插值
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, len(mean_all_cam))
        x_new = np.linspace(0, 1, len(mean_all_test))
        f = interp1d(x_old, mean_all_cam, kind='linear', fill_value='extrapolate')
        mean_all_cam = f(x_new)
    
    # 應用 CAM 到輸入數據
    # 將 CAM 擴展到時間維度
    cam_expanded = np.tile(mean_all_cam[:, np.newaxis], (1, test_all_data.shape[1]))
    hyb_all = test_all_data * cam_expanded
    mean_hyb_all = np.mean(hyb_all, axis=1)
    
    # 創建 EEG info
    info = create_eeg_info(n_channels=test_all_data.shape[0], sfreq=500.0)
    
    # 創建 Evoked 對象
    evoked = mne.EvokedArray(test_all_data, info)
    
    # 繪製可視化
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    
    # 上圖：原始數據的平均拓撲圖
    ax1 = axes[0]
    im1, cn1 = mne.viz.plot_topomap(
        mean_all_test, 
        evoked.info, 
        show=False, 
        axes=ax1, 
        res=1200,
        cmap='RdBu_r'
    )
    class_names = ['AD', 'FTD', 'CN']
    ax1.set_title(f'Average EEG Topography - Class {TARGET_CATEGORY} ({class_names[TARGET_CATEGORY]})', fontsize=12, pad=20)
    plt.colorbar(im1, ax=ax1)
    
    # 下圖：應用 CAM 後的拓撲圖
    ax2 = axes[1]
    im2, cn2 = mne.viz.plot_topomap(
        mean_hyb_all, 
        evoked.info, 
        show=False, 
        axes=ax2, 
        res=1200,
        cmap='RdBu_r'
    )
    ax2.set_title(f'Class Activation Topography (CAT) - Class {TARGET_CATEGORY} ({class_names[TARGET_CATEGORY]})', fontsize=12, pad=20)
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    
    # 保存圖片
    output_path = os.path.join(OUTPUT_DIR, f'CAT_class_{TARGET_CATEGORY}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可視化結果已保存: {output_path}")
    
    plt.close()
    
    print("="*60)
    print("CAT 可視化完成！")
    print("="*60)

if __name__ == "__main__":
    main()



