"""
繪製 EEG 拓撲圖
使用 MNE 繪製 EEG 信號的空間分布

適配 EEG Conformer 實驗：
- 支援 19 通道 EEG 數據
- 3 類別分類（AD/FTD/CN）
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
import mne

# ==================== 配置參數 ====================
# 數據路徑
DATA_PATH = None  # 如果為 None，將從評估結果載入
# 或者指定數據文件路徑，例如：
# DATA_PATH = './results/test_data.npy'
LABEL_PATH = './results/test_labels.npy'  # 標籤路徑

# 可視化參數
TARGET_CLASS = 0  # 要可視化的類別：0=AD, 1=FTD, 2=CN
OUTPUT_DIR = './results/visualization'  # 輸出目錄
N_CHANNELS = 19  # EEG 通道數
SFREQ = 500.0  # 採樣率

# ==================== 數據載入函數 ====================
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
        print("警告: 找不到標籤文件，將處理所有數據")
        labels = None
    
    return data, labels

# ==================== 創建 EEG Info ====================
def create_eeg_info(n_channels=19, sfreq=500.0):
    """創建 19 通道 EEG 的 MNE info 對象"""
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
    """主函數：繪製 EEG 拓撲圖"""
    
    # 創建輸出目錄
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"創建輸出目錄: {OUTPUT_DIR}")
    
    # 載入數據
    print("="*60)
    print("載入數據")
    print("="*60)
    data, labels = load_data()
    
    if data is None:
        return
    
    # 處理數據格式
    # 數據可能是 (trials, 1, channels, timepoints) 或 (trials, channels, timepoints)
    if len(data.shape) == 4:
        # (trials, 1, channels, timepoints) -> (trials, channels, timepoints)
        data = np.squeeze(data, axis=1)
    elif len(data.shape) == 3:
        # 已經是 (trials, channels, timepoints)
        pass
    else:
        raise ValueError(f"不支持的數據形狀: {data.shape}")
    
    print(f"數據形狀: {data.shape}")
    
    # 如果提供了標籤，選擇特定類別的數據
    if labels is not None:
        idx = np.where(labels == TARGET_CLASS)[0]
        if len(idx) == 0:
            print(f"警告: 找不到類別 {TARGET_CLASS} 的數據")
            return
        data_draw = data[idx]
        print(f"選擇類別 {TARGET_CLASS} 的數據: {len(data_draw)} 個樣本")
    else:
        data_draw = data
        print("處理所有數據")
    
    # 計算平均值
    # data_draw shape: (trials, channels, timepoints)
    mean_trial = np.mean(data_draw, axis=0)  # (channels, timepoints) - 所有 trial 的平均
    
    # 標準化（可選）
    # mean_trial = (mean_trial - np.mean(mean_trial)) / np.std(mean_trial)
    
    # 計算每個通道的平均值（對時間維度求平均）
    mean_ch = np.mean(mean_trial, axis=1)  # (channels,) - 每個通道的平均值
    
    # 創建 EEG info
    info = create_eeg_info(n_channels=data.shape[1], sfreq=SFREQ)
    
    # 創建 Evoked 對象
    # mean_trial shape: (channels, timepoints)
    evoked = mne.EvokedArray(mean_trial, info)
    
    # 繪製拓撲圖
    plt.figure(figsize=(8, 6))
    im, cn = mne.viz.plot_topomap(
        mean_ch, 
        evoked.info, 
        show=False,
        cmap='RdBu_r',
        res=1200
    )
    plt.colorbar(im, label='Amplitude (μV)')
    
    class_names = ['AD', 'FTD', 'CN']
    plt.title(f'EEG Topography - {class_names[TARGET_CLASS]} (Average)', fontsize=14, pad=20)
    
    # 保存圖片
    output_path = os.path.join(OUTPUT_DIR, f'topography_class_{TARGET_CLASS}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"拓撲圖已保存: {output_path}")
    
    plt.close()
    
    print("="*60)
    print("拓撲圖繪製完成！")
    print("="*60)

if __name__ == "__main__":
    main()

