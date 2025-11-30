# 可視化工具使用說明

本目錄包含三個可視化工具，已適配 EEG Conformer 實驗設置。

## 📋 工具列表

1. **CAT.py** - Class Activation Topography (類別激活拓撲圖)
2. **topography.py** - EEG 拓撲圖
3. **tSNE.py** - t-SNE 降維可視化

## 🚀 使用前準備

### 1. 運行評估腳本生成數據

在運行可視化工具之前，請先運行評估腳本生成測試數據：

```bash
python evaluate_conformer_model.py
```

這會生成以下文件：
- `./results/test_data.npy` - 測試數據
- `./results/test_labels.npy` - 測試標籤

### 2. 確保模型文件存在

確保訓練好的模型文件存在：
- `./checkpoints/best_model.pth` - 最佳模型（推薦）
- 或 `./checkpoints/final_model.pth` - 最終模型

## 📊 使用方法

### 1. Class Activation Topography (CAT)

可視化模型關注的腦區：

```bash
python visualization/CAT.py
```

**功能：**
- 使用 GradCAM 生成類別激活圖
- 結合 EEG 拓撲圖顯示模型關注的腦區
- 生成兩個拓撲圖：原始信號和應用 CAM 後的信號

**配置參數（在 CAT.py 中修改）：**
- `MODEL_PATH`: 模型路徑（預設：`./checkpoints/best_model.pth`）
- `TARGET_CATEGORY`: 要可視化的類別（0=AD, 1=FTD, 2=CN）
- `N_SAMPLES`: 要處理的樣本數（None 表示處理所有樣本）
- `OUTPUT_DIR`: 輸出目錄（預設：`./results/visualization`）

**輸出：**
- `./results/visualization/CAT_class_{類別}.png`

### 2. EEG 拓撲圖

繪製 EEG 信號的空間分布：

```bash
python visualization/topography.py
```

**功能：**
- 繪製特定類別的平均 EEG 信號拓撲圖
- 顯示信號在頭皮上的空間分布

**配置參數（在 topography.py 中修改）：**
- `TARGET_CLASS`: 要可視化的類別（0=AD, 1=FTD, 2=CN）
- `OUTPUT_DIR`: 輸出目錄（預設：`./results/visualization`）

**輸出：**
- `./results/visualization/topography_class_{類別}.png`

### 3. t-SNE 降維可視化

可視化模型學習到的特徵空間：

```bash
python visualization/tSNE.py
```

**功能：**
- 從模型提取特徵
- 使用 t-SNE 降維到 2D
- 可視化不同類別在特徵空間中的分布

**配置參數（在 tSNE.py 中修改）：**
- `MODEL_PATH`: 模型路徑（預設：`./checkpoints/best_model.pth`）
- `PERPLEXITY`: t-SNE 的 perplexity 參數（預設：30）
- `OUTPUT_DIR`: 輸出目錄（預設：`./results/visualization`）

**輸出：**
- `./results/visualization/tsne_visualization.png`

## ⚙️ 配置說明

### 模型參數

所有工具都使用以下模型參數（與訓練時一致）：
- `EMB_SIZE = 40` - Transformer 嵌入維度
- `DEPTH = 6` - Transformer 編碼器層數
- `N_CLASSES = 3` - 分類類別數（AD/FTD/CN）

### 數據格式

- **輸入數據格式**: `(trials, 1, 19, 1000)` 或 `(trials, 19, 1000)`
  - `trials`: 試次數
  - `1`: 卷積通道維度
  - `19`: EEG 通道數
  - `1000`: 時間點數（2秒 @ 500Hz）

- **標籤格式**: `(trials,)` - 整數數組，值為 0, 1, 或 2
  - `0`: AD (阿茲海默症)
  - `1`: FTD (額顳葉失智症)
  - `2`: CN (對照組)

### EEG 通道映射

使用標準 10-20 系統的 19 通道：
- Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2
- F7, F8, T3, T4, T5, T6
- Fz, Cz, Pz

## 🔧 自定義數據路徑

如果您的數據不在預設位置，可以在各腳本中修改：

```python
# 在 CAT.py, topography.py, tSNE.py 中
DATA_PATH = '/path/to/your/data.npy'  # 指定數據路徑
LABEL_PATH = '/path/to/your/labels.npy'  # 指定標籤路徑
```

## 📝 注意事項

1. **GPU 設定**: 預設使用 GPU 0，可在腳本中修改 `gpus = [0]`
2. **記憶體**: 如果數據量很大，可以設置 `N_SAMPLES` 限制處理的樣本數
3. **輸出目錄**: 所有可視化結果保存在 `./results/visualization/` 目錄
4. **依賴套件**: 確保已安裝所有必要的套件（mne, matplotlib, sklearn, einops 等）

## 🐛 常見問題

### Q: 找不到數據文件
A: 請先運行 `evaluate_conformer_model.py` 生成測試數據，或手動指定數據路徑。

### Q: 模型載入失敗
A: 檢查模型路徑是否正確，確保模型文件存在且格式正確。

### Q: 通道數不匹配
A: 確保數據有 19 個通道，與模型訓練時的配置一致。

### Q: 記憶體不足
A: 減少 `N_SAMPLES` 或 `batch_size` 參數，或使用 CPU 模式。

## 📚 參考資料

- [MNE-Python 文檔](https://mne.tools/stable/index.html)
- [GradCAM 原始實現](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/grad_cam)

