# Special Topics in Innovative Integration of Medicine and EECS I (2025 Fall)

[English](README_EN.md) | **繁體中文**

---

本專案為「醫電創新整合專題 I (2025 Fall)」課程專案，主要研究使用腦電圖（EEG）信號進行神經系統疾病分類。專案包含兩種不同的方法：深度學習方法（EEG-Conformer）和傳統機器學習方法（EEG-Hierarchical-baseline）。

## 🎯 專案簡介

本專案旨在開發和比較不同的機器學習方法，用於從 EEG 信號中自動識別神經系統疾病，特別是：
- **阿茲海默症（AD, Alzheimer's Disease）**
- **額顳葉失智症（FTD, Frontotemporal Dementia）**
- **健康對照組（CN, Control）**

專案使用 **ds004504** 數據集，包含 88 個受試者的 EEG 記錄。

## 📁 專案結構

```
Special-Topics-in-Innovative-Integration-of-Medicine-and-EECS-I-2025fall/
├── README.md                          # 本文件
├── EEG-Conformer/                     # 深度學習方法：Conformer 模型
│   ├── conformer.py                   # Conformer 模型核心實現
│   ├── conformer_train_with_svm_data_split.py  # 訓練腳本
│   ├── evaluate_conformer_model.py    # 模型評估腳本
│   ├── visualization/                 # 可視化工具
│   ├── checkpoints-1/                 # 訓練好的模型權重
│   ├── results/                       # 評估結果和可視化
│   └── README.md                      # Conformer 專案說明
├── EEG-Hierarchical-baseline/         # 傳統機器學習方法：階層式分類
│   ├── baseline.ipynb                 # Baseline 分類器（單階段）
│   ├── two-stage-classification.ipynb # 兩階段分類器
│   ├── gsp_feature_extraction.py      # 圖信號處理特徵提取
│   └── features_tv.csv                # 提取的特徵數據
└── asset/                             # 實驗結果和資源
    ├── baseline_results/              # Baseline 方法結果
    │   ├── logistic_regression/       # 邏輯回歸結果
    │   ├── random_forest/             # 隨機森林結果
    │   └── SVM/                       # 支持向量機結果
    └── two_stage_results/             # 兩階段方法結果
        ├── logistic_regression/       # 邏輯回歸結果
        ├── random_forest/             # 隨機森林結果
        └── SVM/                       # 支持向量機結果
```

## 🔬 方法概述

### 1. EEG-Conformer（深度學習方法）

EEG-Conformer 是一個結合卷積神經網路（CNN）和 Transformer 的混合架構，專為 EEG 信號處理設計。

**主要特點：**
- 端到端訓練，無需手動特徵工程
- 結合 CNN 的局部特徵提取和 Transformer 的全局依賴建模
- 提供 Class Activation Topography (CAT) 等可視化工具
- 使用 SVM 數據分割策略（訓練集 30、驗證集 31、測試集 27）

**詳細說明：** 請參閱 [EEG-Conformer/README.md](EEG-Conformer/README.md)

### 2. EEG-Hierarchical-baseline（傳統機器學習方法）

階層式分類方法使用傳統機器學習分類器，結合圖信號處理（GSP）特徵提取。

**主要特點：**
- **單階段分類（Baseline）**：直接進行三類分類（AD/FTD/CN）
- **兩階段分類（Two-stage）**：先區分疾病組（AD+FTD）與對照組（CN），再區分 AD 與 FTD
- 使用圖信號處理提取總變分（Total Variation）等特徵
- 支援多種分類器：邏輯回歸、隨機森林、SVM

**詳細說明：** 請參閱各 Jupyter Notebook 文件

## 📊 數據集

本專案使用 **ds004504** 數據集：
- **總受試者數**：88 人
- **類別分布**：AD、FTD、CN
- **數據格式**：BIDS 格式的 EEG 數據（.set 文件）
- **數據分割**：
  - 訓練集：30 個樣本
  - 驗證集：31 個樣本
  - 測試集：27 個樣本

**重要提示：** 使用本專案或數據集時，請引用相關論文（見引用部分）。

## 🚀 快速開始

### 環境設置

#### 1. 安裝 Python 依賴

```bash
# 安裝 PyTorch（根據您的 CUDA 版本選擇）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安裝其他依賴
pip install numpy pandas mne scipy scikit-learn einops matplotlib seaborn pygsp networkx
```

#### 2. 準備數據

確保數據集路徑正確設置。在相關腳本中修改數據路徑：

```python
BASE_DIR = '/path/to/your/data'
DATASET_DIR = os.path.join(BASE_DIR, 'ds004504')
```

### 運行 EEG-Conformer

```bash
cd EEG-Conformer
python conformer_train_with_svm_data_split.py
```

### 運行 EEG-Hierarchical-baseline

```bash
cd EEG-Hierarchical-baseline
# 使用 Jupyter Notebook 打開並運行
jupyter notebook baseline.ipynb
# 或
jupyter notebook two-stage-classification.ipynb
```

## 📈 實驗結果

實驗結果保存在 `asset/` 目錄下：

- **baseline_results/**：單階段分類結果
  - 包含混淆矩陣、評估指標等
- **two_stage_results/**：兩階段分類結果
  - 包含混淆矩陣、評估指標、預測結果 CSV 等

### 結果文件說明

每個結果目錄包含：
- `confusion_matrix_test.png`：測試集混淆矩陣
- `confusion_matrix_validation.png`：驗證集混淆矩陣
- `evaluation_metrics.txt`：評估指標（準確率、精確率、召回率、F1 分數等）
- `test_predictions.csv` / `validation_predictions.csv`：預測結果（僅兩階段方法）

## 🔧 主要依賴

- **PyTorch**：深度學習框架
- **MNE**：EEG 數據處理
- **scikit-learn**：傳統機器學習方法
- **PyGSP**：圖信號處理
- **NumPy, Pandas**：數據處理
- **Matplotlib, Seaborn**：可視化

## 📝 引用

如果您使用本專案或相關數據集，請引用：

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

### 引用相關資源

```bibtex
@misc{Xmootoo2025,
  author = {Xmootoo},
  title = {Applying the Graph Discrete Fourier Transform to EEG Data for Alzheimer Disease Detection},
  howpublished = {\url{https://github.com/xmootoo/gsp-alzheimer-detection}},
  note = {Accessed November 24, 2025},
  year = {2025}
}
```

## 👥 作者



## 🔗 相關資源

- [EEG-Conformer 詳細文檔](EEG-Conformer/README.md)
- [EEG-Conformer 可視化工具說明](EEG-Conformer/visualization/README_visualization.md)
- [ds004504 數據集](https://openneuro.org/datasets/ds004504)

## 📧 聯繫方式

如有問題或建議，請聯繫：[待補充]

---

**最後更新：** 2025年12月
