"""
使用 t-SNE 可視化模型學習到的特徵空間分布

適配 EEG Conformer 實驗：
- 從模型提取特徵
- 使用 t-SNE 降維到 2D
- 可視化不同類別在特徵空間中的分布
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
from sklearn import manifold
from einops import reduce
import torch
import torch.nn as nn

# 導入 Conformer 模型
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from conformer import Conformer

# ==================== 配置參數 ====================
# 模型參數
MODEL_PATH = './checkpoints/best_model.pth'
EMB_SIZE = 40
DEPTH = 6
N_CLASSES = 3

# 數據參數
DATA_PATH = None  # 如果為 None，將從評估結果載入
LABEL_PATH = './results/test_labels.npy'

# t-SNE 參數
PERPLEXITY = 30  # t-SNE 的 perplexity 參數
RANDOM_STATE = 166
OUTPUT_DIR = './results/visualization'

# 設備設定
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# ==================== 數據載入 ====================
def load_data():
    """載入數據"""
    if DATA_PATH is not None and os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH)
        print(f"從指定路徑載入數據: {data.shape}")
    else:
        # 嘗試從評估結果載入
        test_data_path = './results/test_data.npy'
        if os.path.exists(test_data_path):
            data = np.load(test_data_path)
            print(f"從評估結果載入數據: {data.shape}")
        else:
            print("錯誤: 找不到數據文件，請先運行評估腳本或指定數據路徑")
            return None, None
    
    # 載入標籤
    if os.path.exists(LABEL_PATH):
        labels = np.load(LABEL_PATH)
        print(f"載入標籤: {labels.shape}")
    else:
        print("錯誤: 找不到標籤文件")
        return None, None
    
    return data, labels

# ==================== 模型載入 ====================
def load_model(model_path, emb_size=40, depth=6, n_classes=3):
    """載入訓練好的 Conformer 模型"""
    model = Conformer(emb_size=emb_size, depth=depth, n_classes=n_classes)
    
    if os.path.exists(model_path):
        print(f"載入模型: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        
        # 處理 DataParallel 包裝的模型
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("模型載入成功")
        return model
    else:
        raise FileNotFoundError(f"找不到模型文件: {model_path}")

# ==================== 提取特徵 ====================
def extract_features(model, data, batch_size=32):
    """從模型提取特徵"""
    model.eval()
    features_list = []
    
    # 確保數據格式正確
    if len(data.shape) == 3:
        # (trials, channels, timepoints) -> (trials, 1, channels, timepoints)
        data = np.expand_dims(data, axis=1)
    elif len(data.shape) == 4:
        # 已經是 (trials, 1, channels, timepoints)
        pass
    else:
        raise ValueError(f"不支持的數據形狀: {data.shape}")
    
    print(f"提取特徵，數據形狀: {data.shape}")
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch_data).to(device)
            
            # 前向傳播獲取特徵
            # Conformer 的 forward 返回 (features, output)
            features, _ = model(batch_tensor)
            features_list.append(features.cpu())
            
            if (i + batch_size) % 100 == 0:
                print(f"  處理進度: {min(i+batch_size, len(data))}/{len(data)}")
    
    # 合併所有特徵
    all_features = torch.cat(features_list, dim=0)
    print(f"特徵形狀: {all_features.shape}")
    
    return all_features

# ==================== t-SNE 可視化 ====================
def plt_tsne(features, labels, perplexity=30, output_path='./results/visualization/tsne.png'):
    """
    使用 t-SNE 可視化特徵分布
    
    Parameters
    ----------
    features : torch.Tensor or np.ndarray
        特徵張量，形狀為 (n_samples, feature_dim)
    labels : np.ndarray
        標籤數組，形狀為 (n_samples,)
    perplexity : int
        t-SNE 的 perplexity 參數
    output_path : str
        輸出圖片路徑
    """
    # 轉換為 numpy
    if isinstance(features, torch.Tensor):
        features = features.cpu().detach().numpy()
    
    # 如果特徵是多維的，需要降維
    if len(features.shape) > 2:
        # 使用 einops 對空間維度求平均
        features = reduce(torch.tensor(features), 'b n e -> b e', reduction='mean').numpy()
    
    labels = np.array(labels)
    
    print(f"特徵形狀: {features.shape}")
    print(f"標籤形狀: {labels.shape}")
    print(f"標籤分布: {np.bincount(labels.astype(int))}")
    
    # 執行 t-SNE
    print("執行 t-SNE 降維...")
    tsne = manifold.TSNE(
        n_components=2, 
        perplexity=perplexity, 
        init='pca', 
        random_state=RANDOM_STATE,
        max_iter=1000  # 在較新版本的 scikit-learn 中使用 max_iter 而不是 n_iter
    )
    X_tsne = tsne.fit_transform(features)
    
    # 歸一化到 [0, 1]
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    
    # 繪製可視化
    plt.figure(figsize=(10, 8))
    
    # 類別名稱和顏色
    class_names = ['AD', 'FTD', 'CN']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 藍色、橙色、綠色
    
    # 為每個類別繪製散點
    for class_id in range(N_CLASSES):
        mask = labels == class_id
        if np.any(mask):
            plt.scatter(
                X_norm[mask, 0], 
                X_norm[mask, 1], 
                c=colors[class_id],
                label=class_names[class_id],
                alpha=0.6,
                s=50
            )
    
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=12, loc='best')
    plt.title('t-SNE visualization', fontsize=14, pad=20)
    plt.tight_layout()
    
    # 保存圖片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"t-SNE 可視化已保存: {output_path}")
    
    plt.close()

# ==================== 主函數 ====================
def main():
    """主函數：執行 t-SNE 可視化"""
    
    # 創建輸出目錄
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"創建輸出目錄: {OUTPUT_DIR}")
    
    # 載入數據
    print("="*60)
    print("步驟 1: 載入數據")
    print("="*60)
    data, labels = load_data()
    
    if data is None or labels is None:
        return
    
    # 載入模型
    print("="*60)
    print("步驟 2: 載入模型")
    print("="*60)
    model = load_model(MODEL_PATH, EMB_SIZE, DEPTH, N_CLASSES)
    
    # 提取特徵
    print("="*60)
    print("步驟 3: 提取特徵")
    print("="*60)
    features = extract_features(model, data, batch_size=32)
    
    # 執行 t-SNE 可視化
    print("="*60)
    print("步驟 4: t-SNE 可視化")
    print("="*60)
    output_path = os.path.join(OUTPUT_DIR, 'tsne_visualization.png')
    plt_tsne(features, labels, perplexity=PERPLEXITY, output_path=output_path)
    
    print("="*60)
    print("t-SNE 可視化完成！")
    print("="*60)

if __name__ == "__main__":
    main()
