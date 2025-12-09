# EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization

本專案實現了 EEG Conformer 模型，這是一個結合卷積神經網路（CNN）和 Transformer 的深度學習架構，用於腦電圖（EEG）信號的解碼和可視化。該模型能夠有效捕捉 EEG 信號的局部特徵和全局依賴關係，在神經系統疾病分類任務中表現優異。

## 🎯 專案簡介

EEG Conformer 是一個專為 EEG 信號處理設計的深度學習模型，主要特點包括：

- **混合架構**：結合 CNN 的局部特徵提取能力和 Transformer 的全局依賴建模能力
- **端到端訓練**：從原始 EEG 時間序列直接進行分類，無需手動特徵工程
- **可視化功能**：提供 Class Activation Topography (CAT) 等可視化工具，幫助理解模型決策
- **靈活的數據處理**：支援多種數據格式和預處理流程

本專案特別針對神經系統疾病分類任務（如阿茲海默症 AD、額顳葉失智症 FTD、對照組 CN）進行了優化，使用 ds004504 數據集（包含 88 個受試者的 EEG 記錄）進行訓練和評估。請注意，使用本專案或數據集時需要引用相關論文（見引用部分）。

## 📁 專案結構

```
EEG-Conformer/
├── conformer.py                          # Conformer 模型核心實現
├── conformer_train_with_svm_data_split.py # 使用 SVM 數據分割策略的訓練腳本
├── evaluate_conformer_model.py            # 模型評估腳本
├── visualization/                        # 可視化工具
│   ├── CAT.py                            # Class Activation Topography 可視化
│   ├── topography.py                     # EEG 拓撲圖可視化
│   ├── tSNE.py                           # t-SNE 降維可視化
│   └── utils.py                          # 可視化工具函數
├── checkpoints/                          # 模型檢查點保存目錄
│   ├── best_model.pth                    # 最佳模型權重
│   ├── final_model.pth                   # 最終模型權重
│   └── checkpoint.pth                    # 訓練檢查點
├── results/                              # 結果輸出目錄
│   ├── evaluation_report.txt             # 評估報告
│   ├── confusion_matrix_*.png            # 混淆矩陣圖
│   ├── roc_curves_*.png                  # ROC 曲線圖
│   ├── class_distribution_*.png          # 類別分布圖
│   └── *.npy                             # 預測結果和概率
├── README.md                             # 本文件
└── LICENSE                                # 許可證文件
```

## 🔧 環境設置

### 系統要求

- Python 3.7+
- CUDA 支援的 GPU（建議至少 8GB 顯存）
- Linux 或 macOS 系統

### 安裝依賴

```bash
# 安裝 PyTorch（根據您的 CUDA 版本選擇）
# 例如，CUDA 11.8：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install numpy pandas mne scipy scikit-learn einops matplotlib seaborn
```

### 完整依賴列表

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
mne>=0.24.0
scipy>=1.7.0
scikit-learn>=1.0.0
einops>=0.4.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📊 數據準備

### 數據集

本專案使用 **ds004504** 數據集進行訓練和評估。該數據集包含 88 個受試者的 EEG 記錄，涵蓋阿茲海默症（AD）、額顳葉失智症（FTD）和健康對照組（CN）三種類別。

**數據集引用**：如果您使用本專案或 ds004504 數據集，請引用原始數據集論文（見下方引用部分）。

### 數據集格式

本專案支援 BIDS 格式的 EEG 數據集（如 ds004504）。數據應組織如下：

```
ds004504/
├── participants.tsv                      # 參與者信息和標籤
├── sub-001/
│   └── eeg/
│       └── sub-001_task-rest_eeg.set     # EEG 數據文件
├── sub-002/
│   └── eeg/
│       └── sub-002_task-rest_eeg.set
└── ...
```

### 數據要求

- **採樣率**：原始採樣率（將自動降採樣到 500 Hz）
- **通道數**：至少 19 個 EEG 通道
- **時間長度**：每個 epoch 將被分割為 2 秒的窗口
- **標籤格式**：在 `participants.tsv` 中，群組標籤應為 'A'（AD）、'F'（FTD）或 'C'（CN）

### 數據預處理

數據預處理流程包括：

1. **降採樣**：將原始 EEG 數據降採樣到 500 Hz
2. **通道選擇**：選擇 19 個標準 EEG 通道
3. **Epoch 創建**：將連續數據分割為 2 秒的 epochs（1000 個時間點）
4. **標準化**：對每個通道分別進行標準化（零均值、單位方差）
5. **數據分割**：按照指定策略分割訓練集、驗證集和測試集

## 🚀 使用方法

### 1. 訓練模型

#### 使用 SVM 數據分割策略訓練

```bash
python conformer_train_with_svm_data_split.py
```

在腳本中修改以下參數以適應您的環境：

```python
BASE_DIR = '/path/to/your/eeg/data'
DATASET_DIR = os.path.join(BASE_DIR, 'ds004504')
```

#### 訓練參數配置

在 `ConformerTrainer` 初始化時可以調整以下參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 2 | 批次大小 |
| `n_epochs` | 2000 | 訓練輪數 |
| `lr` | 0.0002 | 學習率 |
| `emb_size` | 40 | Transformer 嵌入維度 |
| `depth` | 6 | Transformer 編碼器層數 |
| `n_classes` | 3 | 分類類別數（AD/FTD/CN） |
| `window_length` | 2.0 | 時間窗口長度（秒） |
| `target_sfreq` | 500 | 目標採樣率（Hz） |
| `n_channels` | 19 | EEG 通道數 |
| `target_timepoints` | 1000 | 目標時間點數 |

#### 數據分割策略

預設使用與 SVM 分類相同的數據分割策略：
- **訓練集**：30 個受試者
- **驗證集**：31 個受試者
- **測試集**：27 個受試者

### 2. 評估模型

訓練完成後，使用評估腳本進行完整評估：

```bash
python evaluate_conformer_model.py
```

評估腳本會：
- 載入最佳模型或指定模型
- 在驗證集和測試集上進行評估
- 計算多種評估指標（準確率、精確率、召回率、F1 分數、AUC 等）
- 生成混淆矩陣、ROC 曲線、類別分布圖等可視化結果
- 保存預測結果和概率到 `.npy` 文件

### 3. 可視化

#### Class Activation Topography (CAT)

使用 CAT 可視化模型關注的腦區：

```bash
python visualization/CAT.py
```

#### EEG 拓撲圖

繪製 EEG 信號的空間分布：

```bash
python visualization/topography.py
```

#### t-SNE 降維可視化

可視化模型學習到的特徵空間：

```bash
python visualization/tSNE.py
```

## 🏗️ 模型架構

### Conformer 架構概述

EEG Conformer 由三個主要組件構成：

1. **Patch Embedding（補丁嵌入）**
   - 使用淺層 CNN 提取局部時空特徵
   - 將 EEG 信號轉換為序列化的補丁表示
   - 包含兩個卷積層和池化操作

2. **Transformer Encoder（Transformer 編碼器）**
   - 多層 Transformer 編碼器塊
   - 使用多頭自注意力機制捕捉全局依賴
   - 包含殘差連接和層歸一化

3. **Classification Head（分類頭）**
   - 全局平均池化
   - 全連接層進行最終分類

### 詳細架構

```
輸入: (batch_size, 1, 19, 1000)
    ↓
Patch Embedding
    ├─ Conv2d(1→40, kernel=(1,25))
    ├─ Conv2d(40→40, kernel=(19,1))
    ├─ BatchNorm2d + ELU
    ├─ AvgPool2d(kernel=(1,75), stride=(1,15))
    └─ Dropout(0.5)
    ↓
Transformer Encoder (depth=6)
    ├─ Multi-Head Attention (10 heads)
    ├─ Feed Forward Network
    └─ Residual Connections
    ↓
Classification Head
    ├─ Global Average Pooling
    ├─ LayerNorm
    └─ Linear(emb_size → n_classes)
    ↓
輸出: (batch_size, n_classes)
```

### 關鍵設計特點

1. **局部特徵提取**：CNN 層捕捉 EEG 信號的局部時空模式
2. **全局依賴建模**：Transformer 編碼器學習長距離依賴關係
3. **數據增強**：使用 Segmentation and Reconstruction (S&R) 方法進行數據增強
4. **正則化**：使用 Dropout 和 Batch Normalization 防止過擬合

## 📈 評估指標

模型評估包括以下指標：

### 分類指標

- **準確率（Accuracy）**：整體分類正確率
- **平衡準確率（Balanced Accuracy）**：考慮類別不平衡的準確率
- **精確率（Precision）**：每個類別的精確率
- **召回率（Recall）**：每個類別的召回率
- **F1 分數（F1-Score）**：精確率和召回率的調和平均
- **AUC-ROC**：受試者工作特徵曲線下面積

### 可視化輸出

- **混淆矩陣**：展示分類結果的詳細分布
- **ROC 曲線**：多類別 ROC 曲線（一對多策略）
- **類別分布**：訓練集、驗證集和測試集的類別分布

## 📊 結果說明

### 輸出文件

訓練和評估過程會生成以下文件：

#### 模型文件（`checkpoints/`）

- `best_model.pth`：驗證集上表現最好的模型權重
- `final_model.pth`：訓練完成後的最終模型權重
- `checkpoint.pth`：訓練檢查點（包含優化器狀態等）

#### 結果文件（`results/`）

- `evaluation_report.txt`：詳細的評估報告
- `log_conformer_svm_split.txt`：訓練過程日誌
- `confusion_matrix_validation.png`：驗證集混淆矩陣
- `confusion_matrix_test.png`：測試集混淆矩陣
- `roc_curves_validation.png`：驗證集 ROC 曲線
- `roc_curves_test.png`：測試集 ROC 曲線
- `class_distribution_*.png`：類別分布圖
- `*_predictions.npy`：預測標籤
- `*_probabilities.npy`：預測概率
- `*_true_labels.npy`：真實標籤

### 結果解讀

1. **混淆矩陣**：對角線元素表示正確分類的樣本數，非對角線元素表示誤分類
2. **ROC 曲線**：曲線越靠近左上角，分類性能越好
3. **類別分布**：檢查數據集是否存在類別不平衡問題

## 🔍 故障排除

### 常見問題

#### 1. 找不到 `.set` 文件

**問題**：訓練時提示找不到 EEG 數據文件

**解決方案**：
- 檢查數據路徑是否正確（`BASE_DIR` 和 `DATASET_DIR`）
- 確認 `derivatives` 資料夾或原始數據資料夾存在
- 確認文件命名格式符合 BIDS 規範

#### 2. 通道數不匹配

**問題**：提示通道數不符合預期

**解決方案**：
- 腳本會自動處理通道數差異
- 如果通道數不足 19，該受試者會被自動跳過
- 可以在代碼中調整 `n_channels` 參數以適應您的數據

#### 3. GPU 記憶體不足

**問題**：訓練時出現 CUDA out of memory 錯誤

**解決方案**：
- 減少 `batch_size`（例如從 4 改為 2）
- 減少 `n_epochs` 進行測試
- 使用更小的 `window_length` 或 `target_timepoints`
- 啟用梯度累積（`gradient_accumulation_steps`）

#### 4. 訓練速度慢

**問題**：訓練過程非常緩慢

**解決方案**：
- 確認使用 GPU 而非 CPU
- 檢查 CUDA 是否正確安裝
- 減少數據量或使用更小的模型參數
- 使用混合精度訓練（需要額外實現）

#### 5. 模型性能不佳

**問題**：驗證集或測試集準確率較低

**解決方案**：
- 檢查數據預處理是否正確
- 確認標籤編碼正確（A/F/C → 0/1/2）
- 嘗試調整學習率或增加訓練輪數
- 檢查是否存在類別不平衡問題
- 考慮使用數據增強或調整模型架構

## 🔬 進階使用

### 自定義數據集

要使用自己的數據集，需要：

1. 實現數據載入函數，返回格式為 `(data, labels)`
   - `data`: shape `(n_samples, 1, n_channels, n_timepoints)`
   - `labels`: shape `(n_samples,)`

2. 修改 `ConformerTrainer` 類中的 `load_data` 方法

3. 調整模型參數以適應您的數據維度

### 模型微調

可以通過以下方式微調模型：

1. **調整架構參數**：
   - `emb_size`：嵌入維度（建議範圍：32-64）
   - `depth`：Transformer 層數（建議範圍：4-8）
   - `num_heads`：注意力頭數（建議範圍：8-16）

2. **調整訓練參數**：
   - `lr`：學習率（建議範圍：1e-5 到 1e-3）
   - `batch_size`：批次大小（根據 GPU 記憶體調整）
   - `n_epochs`：訓練輪數

3. **數據增強**：
   - 調整 S&R 數據增強參數
   - 添加其他數據增強方法（如時間扭曲、通道丟失等）

### 遷移學習

可以載入預訓練模型進行遷移學習：

```python
# 載入預訓練模型
model = Conformer(emb_size=40, depth=6, n_classes=3)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# 凍結部分層（可選）
for param in model[0].parameters():  # 凍結 Patch Embedding
    param.requires_grad = False

# 繼續訓練或微調
```

## 📝 開發日誌

### 版本歷史

- **v1.0**：初始版本，實現基本 Conformer 架構
- **v1.1**：添加 SVM 數據分割策略支援
- **v1.2**：完善評估和可視化功能
- **v1.3**：添加檢查點恢復功能

### 待實現功能

- [ ] 混合精度訓練支援
- [ ] 更多數據增強方法
- [ ] 交叉驗證支援
- [ ] 模型解釋性工具
- [ ] 分散式訓練支援

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 許可證

請查看 `LICENSE` 文件了解詳細許可證信息。

## 🙏 致謝

- 感謝原始論文作者 Song et al. 提供的優秀工作
- 感謝所有開源社區的貢獻者
- 本專案參考了多個開源項目，特別感謝：
  - [einops](https://github.com/arogozhnikov/einops) - 張量操作庫
  - [MNE-Python](https://mne.tools/) - EEG/MEG 數據處理庫
  - [PyTorch](https://pytorch.org/) - 深度學習框架

## 📧 聯繫方式

如有問題或建議，請通過以下方式聯繫：

- 提交 GitHub Issue
- 發送郵件至項目維護者

## 🔗 相關資源

- [原始論文](https://ieeexplore.ieee.org/document/10000000)
- [BIDS 格式文檔](https://bids.neuroimaging.io/)
- [MNE-Python 文檔](https://mne.tools/stable/index.html)
- [PyTorch 文檔](https://pytorch.org/docs/stable/index.html)

---

**注意**：本專案僅供研究和教育用途。在臨床應用中，請確保遵循相關法規和倫理準則。


### 引用原始論文

本專案基於以下論文實現：

```bibtex
@article{song2023eeg,
  title = {{EEG Conformer}: {{Convolutional Transformer}} for {{EEG Decoding}} and {{Visualization}}},
  shorttitle = {{EEG Conformer}},
  author = {Song, Yonghao and Zheng, Qingqing and Liu, Bingchuan and Gao, Xiaorong},
  year = {2023},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume = {31},
  pages = {710--719},
  issn = {1558-0210},
  doi = {10.1109/TNSRE.2022.3230250}
}
```

### 引用數據集

本專案使用 ds004504 數據集，請引用以下數據集論文：

```bibtex
@dataset{miltiadous2023eeg,
  title = {A dataset of 88 EEG recordings from: Alzheimer's disease, Frontotemporal dementia and Healthy subjects},
  author = {Miltiadous, Andreas and Tzimourta, Katerina D. and Afrantou, Theodora and Ioannidis, Panagiotis and Grigoriadis, Nikolaos and Tsalikakis, Dimitrios G. and Angelidis, Pantelis and Tsipouras, Markos G. and Glavas, Evripidis and Giannakeas, Nikolaos and Tzallas, Alexandros T.},
  year = {2023},
  publisher = {OpenNeuro},
  type = {Dataset},
  doi = {10.18112/openneuro.ds004504.v1.0.1},
  url = {https://openneuro.org/datasets/ds004504}
}
```

