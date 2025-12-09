# Conformer 訓練腳本（使用 SVM 資料分割策略）

此腳本參考您提供的 SVM 分類程式的資料處理流程，使用相同的資料分割策略來訓練 Conformer 模型。

## 主要特點

1. **相同的資料分割策略**：訓練集 30, 驗證集 31, 測試集 27（總共 88 個樣本）
2. **從原始 EEG 數據載入**：直接從 `.set` 文件讀取 EEG 數據
3. **相同的標籤處理**：從 `participants.tsv` 讀取群組標籤（A/F/C）
4. **標準化處理**：對每個通道分別進行標準化
5. **數據增強**：使用 Segmentation and Reconstruction (S&R) 方法

## 使用方法

### 1. 確保環境設置

確保已安裝必要的套件：
```bash
pip install torch torchvision numpy pandas mne scipy scikit-learn einops
```

### 2. 設置數據路徑

在 `main()` 函數中修改數據路徑：
```python
BASE_DIR = '/ibmnas/427/bachelors/b12901077/eeg'
DATASET_DIR = os.path.join(BASE_DIR, 'ds004504')
```

### 3. 運行訓練

```bash
cd EEG-Conformer
python conformer_train_with_svm_data_split.py
```

## 參數說明

在 `ConformerTrainer` 初始化時可以調整以下參數：

- `base_dir`: 數據根目錄
- `dataset_dir`: 數據集目錄（預設為 `base_dir/ds004504`）
- `checkpoint_dir`: 模型檢查點保存目錄（預設為 `./checkpoints`）
- `batch_size`: 批次大小（預設為 4）
- `n_epochs`: 訓練輪數（預設為 2000）
- `lr`: 學習率（預設為 0.0002）
- `emb_size`: 嵌入大小（預設為 40）
- `depth`: Transformer 深度（預設為 6）
- `n_classes`: 類別數（預設為 3：AD/FTD/CN）
- `window_length`: 時間視窗長度，秒（預設為 2.0）
- `target_sfreq`: 目標採樣率（預設為 500 Hz）
- `n_channels`: EEG 通道數（預設為 19）
- `target_timepoints`: 目標時間點數（預設為 1000）

## 輸出文件

訓練過程中會生成以下文件：

1. `./checkpoints/best_model.pth`: 最佳模型權重
2. `./checkpoints/final_model.pth`: 最終模型權重
3. `./results/log_conformer_svm_split.txt`: 訓練日誌

## 數據處理流程

1. **數據載入**：從 `ds004504` 數據集載入所有受試者的 EEG 數據
2. **標籤獲取**：從 `participants.tsv` 讀取群組標籤
3. **數據預處理**：
   - 降採樣到目標採樣率（500 Hz）
   - 選擇 19 個 EEG 通道
   - 創建固定長度的 epochs（2 秒 = 1000 個時間點）
4. **標準化**：對每個通道分別進行標準化
5. **數據分割**：使用與 SVM 相同的分割策略（30/31/27）
6. **標籤編碼**：將 A/F/C 編碼為 0/1/2

## 與 SVM 分類程式的對應關係

| SVM 程式 | Conformer 訓練腳本 |
|---------|-------------------|
| 從 `features_tv.csv` 讀取特徵 | 從原始 `.set` 文件讀取 EEG 數據 |
| 使用 9 個特徵 | 使用原始時間序列（19 通道 × 1000 時間點）|
| 訓練集 30, 驗證集 31, 測試集 27 | 相同的分割策略 |
| `RobustScaler` 標準化 | 對每個通道分別標準化 |
| `LabelEncoder` 編碼標籤 | 相同的標籤編碼方式 |

## 注意事項

1. 確保數據路徑正確，特別是 `base_dir` 和 `dataset_dir`
2. 確保有足夠的 GPU 記憶體（建議至少 8GB）
3. 訓練時間取決於數據量和硬體配置，可能需要數小時
4. 如果某個受試者的數據無法載入，腳本會跳過並繼續處理其他受試者

## 故障排除

### 問題：找不到 `.set` 文件
- 檢查數據路徑是否正確
- 確認 `derivatives` 資料夾或原始數據資料夾存在

### 問題：通道數不匹配
- 腳本會自動處理通道數差異
- 如果通道數不足 19，該受試者會被跳過

### 問題：記憶體不足
- 減少 `batch_size`
- 減少 `n_epochs` 進行測試
- 使用更小的 `window_length` 或 `target_timepoints`

## 參考

- 原始 Conformer 實現：`conformer.py`
- SVM 分類程式：您提供的完整階層式分類程式


