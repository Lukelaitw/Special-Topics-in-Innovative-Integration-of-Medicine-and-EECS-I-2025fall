"""
使用訓練好的 Conformer 模型進行評估

此腳本載入訓練好的模型並進行完整的評估，包括：
- 載入最佳模型或指定模型
- 評估驗證集和測試集
- 計算所有評估指標（與訓練時相同）
"""

import os
import sys
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

# 導入 Conformer 模型
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformer import Conformer

# GPU 設定
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    auc
)
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# 導入可視化庫
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']  # 支持中文顯示
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
sns.set_style("whitegrid")


class EEGDataset(Dataset):
    """EEG 數據集類別"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ConformerEvaluator:
    """使用訓練好的 Conformer 模型進行評估"""
    
    def __init__(
        self,
        model_path=None,  # 模型路徑，如果為 None 則使用 best_model.pth
        checkpoint_path=None,  # 檢查點路徑，如果提供則從檢查點載入
        base_dir='/ibmnas/427/bachelors/b12901077/eeg',
        dataset_dir=None,
        checkpoint_dir='./checkpoints',
        batch_size=4,
        emb_size=40,
        depth=6,
        n_classes=3,
        window_length=2.0,  # 秒
        target_sfreq=500,   # 目標採樣率
        n_channels=19,      # EEG 通道數
        target_timepoints=1000,  # 目標時間點數
    ):
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.base_dir = base_dir
        if dataset_dir is None:
            self.dataset_dir = os.path.join(base_dir, 'ds004504')
        else:
            self.dataset_dir = dataset_dir
        
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.depth = depth
        self.n_classes = n_classes
        self.window_length = window_length
        self.target_sfreq = target_sfreq
        self.n_channels = n_channels
        self.target_timepoints = target_timepoints
        
        # 創建結果目錄
        if not os.path.exists('./results'):
            os.makedirs('./results')
        
        # 設備設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        
        # 標籤編碼器
        self.le = LabelEncoder()
        
        # 模型
        self.model = Conformer(emb_size=emb_size, depth=depth, n_classes=n_classes)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.to(self.device)
        
        print("="*60)
        print("Conformer 評估器初始化完成")
        print("="*60)
        print(f"設備: {self.device}")
        print(f"批次大小: {self.batch_size}")
        print(f"嵌入大小: {self.emb_size}")
        print(f"深度: {self.depth}")
        print(f"類別數: {self.n_classes}")
        print("="*60)
    
    def load_eeg_data_from_subject(self, subject_id):
        """
        從單個受試者載入 EEG 數據
        """
        subject_id_str = f"sub-{subject_id:03d}"
        
        # 讀取 participants.tsv 獲取標籤
        participants_file = os.path.join(self.dataset_dir, 'participants.tsv')
        participants_df = pd.read_csv(participants_file, delimiter='\t')
        
        # 找到對應的受試者
        subject_row = participants_df[participants_df['participant_id'] == subject_id_str]
        if subject_row.empty:
            print(f"警告: 找不到受試者 {subject_id_str} 的資訊")
            return None, None
        
        group = subject_row['Group'].iloc[0]
        
        # 優先使用 derivatives 資料夾中的預處理資料
        set_file = os.path.join(
            self.dataset_dir,
            'derivatives',
            subject_id_str,
            'eeg',
            f'{subject_id_str}_task-eyesclosed_eeg.set'
        )
        
        # 如果 derivatives 不存在，使用原始資料
        if not os.path.exists(set_file):
            set_file = os.path.join(
                self.dataset_dir,
                subject_id_str,
                'eeg',
                f'{subject_id_str}_task-eyesclosed_eeg.set'
            )
        
        if not os.path.exists(set_file):
            print(f"警告: 找不到文件 {set_file}")
            return None, None
        
        try:
            # 讀取原始資料
            raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
            
            # 降採樣到目標採樣率
            if raw.info['sfreq'] != self.target_sfreq:
                raw.resample(self.target_sfreq, verbose=False)
            
            # 選擇 EEG 通道
            picks = mne.pick_types(
                raw.info,
                meg=False,
                eeg=True,
                eog=False,
                stim=False,
                exclude='bads'
            )
            
            # 檢查通道數
            if len(picks) != self.n_channels:
                if len(picks) > self.n_channels:
                    picks = picks[:self.n_channels]
                else:
                    print(f"錯誤: 受試者 {subject_id} 的通道數不足")
                    return None, None
            
            # 創建固定長度的 epochs
            events = mne.make_fixed_length_events(
                raw,
                duration=self.window_length,
                overlap=0.0
            )
            
            # 創建 epochs
            epochs = mne.Epochs(
                raw,
                events,
                tmin=0,
                tmax=self.window_length - 1/self.target_sfreq,
                picks=picks,
                preload=True,
                baseline=None,
                verbose=False
            )
            
            # 獲取數據
            data = epochs.get_data()  # (trials, channels, time_samples)
            
            # 確保時間點為目標值
            if data.shape[2] != self.target_timepoints:
                if data.shape[2] > self.target_timepoints:
                    data = data[:, :, :self.target_timepoints]
                else:
                    from scipy import signal
                    data_resampled = []
                    for trial in data:
                        resampled = signal.resample(trial, self.target_timepoints, axis=1)
                        data_resampled.append(resampled)
                    data = np.array(data_resampled)
            
            # 確保通道數正確
            if data.shape[1] != self.n_channels:
                print(f"警告: 數據通道數為 {data.shape[1]}，預期為 {self.n_channels}")
                return None, None
            
            return data, group
            
        except Exception as e:
            print(f"錯誤: 處理受試者 {subject_id} 時發生錯誤: {e}")
            return None, None
    
    def load_all_data(self):
        """
        載入所有受試者的數據，並使用與訓練時相同的分割策略
        """
        print("="*60)
        print("步驟 0: 資料載入和預處理")
        print("="*60)
        
        # 讀取 participants.tsv
        participants_file = os.path.join(self.dataset_dir, 'participants.tsv')
        participants_df = pd.read_csv(participants_file, delimiter='\t')
        
        print(f"總受試者數: {len(participants_df)}")
        
        # 載入所有受試者的數據
        all_data_list = []
        all_labels_list = []
        
        for idx, row in participants_df.iterrows():
            subject_id_str = row['participant_id'].replace('sub-', '')
            subject_id = int(subject_id_str)
            
            print(f"載入受試者 {subject_id}...", end=' ')
            data, group = self.load_eeg_data_from_subject(subject_id)
            
            if data is not None and group is not None:
                n_trials = data.shape[0]
                labels = [group] * n_trials
                
                all_data_list.append(data)
                all_labels_list.extend(labels)
                print(f"✓ ({n_trials} 個 trials)")
            else:
                print("✗")
        
        if len(all_data_list) == 0:
            raise ValueError("錯誤: 沒有成功載入任何數據！")
        
        # 合併所有數據
        X = np.concatenate(all_data_list, axis=0)
        y = np.array(all_labels_list)
        
        del all_data_list, all_labels_list
        import gc
        gc.collect()
        
        print(f"\n總樣本數: {len(X)}")
        print(f"數據形狀: {X.shape}")
        print(f"標籤分布: {pd.Series(y).value_counts().to_dict()}")
        
        # 標準化數據（對每個通道分別標準化）
        print("\n標準化數據...")
        X_normalized = np.zeros_like(X)
        for i in range(X.shape[1]):
            channel_data = X[:, i, :]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            if std_val > 0:
                X_normalized[:, i, :] = (channel_data - mean_val) / std_val
            else:
                X_normalized[:, i, :] = channel_data
        
        X = X_normalized
        del X_normalized
        import gc
        gc.collect()
        
        # 編碼標籤（使用與訓練時相同的編碼）
        y_encoded = self.le.fit_transform(y)
        
        print(f"\n類別編碼: {dict(zip(self.le.classes_, range(len(self.le.classes_))))}")
        print(f"標籤分布 (編碼後): {dict(zip(self.le.classes_, np.bincount(y_encoded)))}")
        
        # 數據分割（與訓練時相同：訓練集 30, 驗證集 31, 測試集 27）
        print("\n數據分割（訓練集 30, 驗證集 31, 測試集 27）...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y_encoded, test_size=27/88, random_state=42, stratify=y_encoded
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=31/61, random_state=42, stratify=y_train_val
        )
        
        print("="*60)
        print("數據分割信息")
        print("="*60)
        print(f"總樣本數: {len(X)}")
        print(f"訓練集樣本數: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
        print(f"驗證集樣本數: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
        print(f"測試集樣本數: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
        print(f"\n訓練集類別分布: {dict(zip(self.le.classes_, np.bincount(y_train)))}")
        print(f"驗證集類別分布: {dict(zip(self.le.classes_, np.bincount(y_val)))}")
        print(f"測試集類別分布: {dict(zip(self.le.classes_, np.bincount(y_test)))}")
        print("="*60)
        
        # 轉換為 Conformer 需要的格式: (trials, 1, channels, timepoints)
        X_train = np.expand_dims(X_train, axis=1)
        X_val = np.expand_dims(X_val, axis=1)
        X_test = np.expand_dims(X_test, axis=1)
        
        del X, y, X_train_val, y_train_val
        import gc
        gc.collect()
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def load_model(self):
        """
        載入訓練好的模型
        """
        print("\n" + "="*60)
        print("載入模型")
        print("="*60)
        
        # 優先使用檢查點（如果提供）
        if self.checkpoint_path is not None:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"檢查點文件不存在: {self.checkpoint_path}")
            
            print(f"從檢查點載入模型: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            
            # 載入模型狀態
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # 載入 LabelEncoder 狀態（如果存在）
            if 'label_encoder_classes' in checkpoint and checkpoint['label_encoder_classes'] is not None:
                self.le.classes_ = checkpoint['label_encoder_classes']
                print(f'已載入 LabelEncoder 狀態: {self.le.classes_}')
            
            # 顯示模型信息
            if 'best_acc' in checkpoint:
                print(f'模型最佳準確率: {checkpoint["best_acc"]:.6f}')
            if 'epoch' in checkpoint:
                print(f'訓練輪數: {checkpoint["epoch"] + 1}')
        
        # 否則使用模型文件
        elif self.model_path is not None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            print(f"載入模型: {self.model_path}")
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(torch.load(self.model_path, map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
        # 默認使用 best_model.pth
        else:
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            if not os.path.exists(best_model_path):
                raise FileNotFoundError(f"找不到最佳模型文件: {best_model_path}")
            
            print(f"載入最佳模型: {best_model_path}")
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(torch.load(best_model_path, map_location=self.device))
            else:
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        self.model.eval()
        print("模型載入完成")
        print("="*60)
    
    def evaluate_dataset(self, data_loader, dataset_name="Dataset"):
        """
        評估數據集
        
        Parameters
        ----------
        data_loader : DataLoader
            數據加載器
        dataset_name : str
            數據集名稱（用於顯示）
        
        Returns
        -------
        y_true_np : np.ndarray
            真實標籤
        y_pred_np : np.ndarray
            預測標籤
        y_proba_np : np.ndarray
            預測概率
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (img, label) in enumerate(data_loader):
                img = Variable(img.to(self.device).type(self.Tensor))
                label = Variable(label.to(self.device).type(self.LongTensor))
                
                Tok, Cls = self.model(img)
                y_pred_batch = torch.max(Cls, 1)[1]
                y_proba_batch = F.softmax(Cls, dim=1)
                
                all_preds.append(y_pred_batch.detach().cpu())
                all_labels.append(label.detach().cpu())
                all_probs.append(y_proba_batch.detach().cpu())
                
                del Tok, Cls, img, label, y_pred_batch, y_proba_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 合併所有批次的結果
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_labels)
        y_proba = torch.cat(all_probs)
        
        # 轉換為 numpy
        y_true_np = y_true.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        y_proba_np = y_proba.cpu().numpy()
        
        del all_preds, all_labels, all_probs, y_pred, y_true, y_proba
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return y_true_np, y_pred_np, y_proba_np
    
    def plot_confusion_matrix(self, cm, class_names, dataset_name="Dataset", save_path=None):
        """
        繪製混淆矩陣熱圖
        
        Parameters
        ----------
        cm : np.ndarray
            混淆矩陣
        class_names : list
            類別名稱列表
        dataset_name : str
            數據集名稱
        save_path : str
            保存路徑，如果為 None 則自動生成
        """
        if save_path is None:
            save_path = f'./results/confusion_matrix_{dataset_name.lower()}.png'
        
        # 計算歸一化混淆矩陣
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # 創建圖形
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 原始混淆矩陣
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'Confusion Matrix - {dataset_name}\n(Count)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        
        # 歸一化混淆矩陣
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title(f'Normalized Confusion Matrix - {dataset_name}\n(Percentage)', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混淆矩陣圖已保存到: {save_path}")
    
    def plot_roc_curves(self, y_true, y_proba, class_names, dataset_name="Dataset", save_path=None):
        """
        繪製 ROC 曲線
        
        Parameters
        ----------
        y_true : np.ndarray
            真實標籤
        y_proba : np.ndarray
            預測概率
        class_names : list
            類別名稱列表
        dataset_name : str
            數據集名稱
        save_path : str
            保存路徑，如果為 None 則自動生成
        """
        if save_path is None:
            save_path = f'./results/roc_curves_{dataset_name.lower()}.png'
        
        # 二值化標籤
        binarized = label_binarize(y_true, classes=[0, 1, 2])
        
        # 計算每個類別的 ROC 曲線
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i, class_name in enumerate(class_names):
            fpr[i], tpr[i], _ = roc_curve(binarized[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 計算宏平均 ROC 曲線
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(class_names)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(class_names)
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        macro_roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        roc_auc["macro"] = macro_roc_auc
        
        # 繪製圖形
        plt.figure(figsize=(10, 8))
        
        # 繪製每個類別的 ROC 曲線
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, class_name in enumerate(class_names):
            plt.plot(fpr[i], tpr[i], 
                    color=colors[i % len(colors)], 
                    lw=2, 
                    label=f'{class_name} (AUC = {roc_auc[i]:.4f})')
        
        # 繪製宏平均 ROC 曲線
        plt.plot(fpr["macro"], tpr["macro"],
                color='navy', linestyle='--', lw=2,
                label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})')
        
        # 繪製對角線（隨機分類器）
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {dataset_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC 曲線圖已保存到: {save_path}")
    
    def plot_class_distribution(self, y_true, y_pred, class_names, dataset_name="Dataset", save_path=None):
        """
        繪製類別分布對比圖
        
        Parameters
        ----------
        y_true : np.ndarray
            真實標籤
        y_pred : np.ndarray
            預測標籤
        class_names : list
            類別名稱列表
        dataset_name : str
            數據集名稱
        save_path : str
            保存路徑，如果為 None 則自動生成
        """
        if save_path is None:
            save_path = f'./results/class_distribution_{dataset_name.lower()}.png'
        
        # 計算分布
        true_counts = [np.sum(y_true == i) for i in range(len(class_names))]
        pred_counts = [np.sum(y_pred == i) for i in range(len(class_names))]
        
        # 創建圖形
        x = np.arange(len(class_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, true_counts, width, label='True Labels', color='#2ca02c', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted Labels', color='#1f77b4', alpha=0.8)
        
        # 添加數值標籤
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Class Distribution Comparison - {dataset_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"類別分布圖已保存到: {save_path}")
    
    def print_evaluation_metrics(self, y_true, y_pred, y_proba, dataset_name="Dataset", return_text=False, plot=True):
        """
        打印完整的評估指標
        
        Parameters
        ----------
        y_true : np.ndarray
            真實標籤
        y_pred : np.ndarray
            預測標籤
        y_proba : np.ndarray
            預測概率
        dataset_name : str
            數據集名稱
        return_text : bool
            是否返回文本字符串（用於保存到文件）
        
        Returns
        -------
        dict or tuple
            如果 return_text=False，返回結果字典
            如果 return_text=True，返回 (結果字典, 文本字符串)
        """
        # 獲取類別名稱
        if hasattr(self.le, 'classes_') and len(self.le.classes_) > 0:
            class_names = self.le.classes_
        else:
            class_names = [f'Class {i}' for i in range(self.n_classes)]
        
        # 構建文本輸出
        text_lines = []
        text_lines.append("\n" + "="*60)
        text_lines.append(f"{dataset_name} Metrics for Conformer Model")
        text_lines.append("="*60)
        
        # Balanced Accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        text_lines.append(f"\nBalanced Accuracy: {balanced_acc:.4f}")
        print("\n" + "="*60)
        print(f"{dataset_name} Metrics for Conformer Model")
        print("="*60)
        print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
        
        # 混淆矩陣
        cm = confusion_matrix(y_true, y_pred)
        text_lines.append(f"\n混淆矩陣:")
        text_lines.append(str(cm))
        print(f"\n混淆矩陣:")
        print(cm)
        
        # 歸一化混淆矩陣
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        text_lines.append("\n歸一化混淆矩陣 (百分比):")
        header = "True \\ Pred"
        for name in class_names:
            header += f"\t{name}"
        text_lines.append(header)
        print("\n歸一化混淆矩陣 (百分比):")
        print(header)
        for i, true_class in enumerate(class_names):
            row = f"{true_class}"
            for j in range(len(class_names)):
                row += f"\t{cm_normalized[i, j]:6.2%}"
            text_lines.append(row)
            print(row)
        
        # 分類報告
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
        text_lines.append(f"\n分類報告:")
        text_lines.append(report)
        print(f"\n分類報告:")
        print(report)
        
        # ROC AUC
        binarized = label_binarize(y_true, classes=[0, 1, 2])
        text_lines.append("\n每個類別的 ROC AUC:")
        print("\n每個類別的 ROC AUC:")
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(binarized[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            line = f"  {class_name} (Class {i}): {roc_auc:.4f}"
            text_lines.append(line)
            print(line)
        
        macro_roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        weighted_roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        
        text_lines.append(f"\n宏平均 ROC AUC: {macro_roc_auc:.4f}")
        text_lines.append(f"加權平均 ROC AUC: {weighted_roc_auc:.4f}")
        print(f"\n宏平均 ROC AUC: {macro_roc_auc:.4f}")
        print(f"加權平均 ROC AUC: {weighted_roc_auc:.4f}")
        
        # 類別預測分布分析
        text_lines.append("\n" + "="*60)
        text_lines.append("類別預測分布分析")
        text_lines.append("="*60)
        pred_counts = {class_names[i]: np.sum(y_pred == i) for i in range(len(class_names))}
        true_counts = {class_names[i]: np.sum(y_true == i) for i in range(len(class_names))}
        text_lines.append(f"真實分布: {true_counts}")
        text_lines.append(f"預測分布: {pred_counts}")
        text_lines.append("="*60)
        print("\n" + "="*60)
        print("類別預測分布分析")
        print("="*60)
        print(f"真實分布: {true_counts}")
        print(f"預測分布: {pred_counts}")
        print("="*60)
        
        result_dict = {
            'balanced_accuracy': balanced_acc,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'classification_report': report,
            'macro_roc_auc': macro_roc_auc,
            'weighted_roc_auc': weighted_roc_auc,
            'class_names': class_names
        }
        
        # 繪製圖表
        if plot:
            try:
                self.plot_confusion_matrix(cm, class_names, dataset_name)
                self.plot_roc_curves(y_true, y_proba, class_names, dataset_name)
                self.plot_class_distribution(y_true, y_pred, class_names, dataset_name)
            except Exception as e:
                print(f"警告: 繪圖時發生錯誤: {e}")
        
        if return_text:
            text_output = '\n'.join(text_lines)
            return result_dict, text_output
        else:
            return result_dict
    
    def evaluate(self, evaluate_val=True, evaluate_test=True, save_to_file=True):
        """
        執行完整評估
        
        Parameters
        ----------
        evaluate_val : bool
            是否評估驗證集
        evaluate_test : bool
            是否評估測試集
        save_to_file : bool
            是否保存結果到 txt 文件
        """
        import datetime
        
        # 收集所有文本輸出
        all_text_lines = []
        all_text_lines.append("="*60)
        all_text_lines.append("Conformer 模型評估報告")
        all_text_lines.append("="*60)
        all_text_lines.append(f"評估時間: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.model_path:
            all_text_lines.append(f"模型路徑: {self.model_path}")
        elif self.checkpoint_path:
            all_text_lines.append(f"檢查點路徑: {self.checkpoint_path}")
        else:
            all_text_lines.append(f"使用默認模型: ./checkpoints/best_model.pth")
        all_text_lines.append("="*60)
        
        # 載入模型
        self.load_model()
        all_text_lines.append("\n模型載入完成")
        
        # 載入數據
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_all_data()
        
        # 保存原始測試數據供可視化使用
        if save_to_file:
            np.save('./results/test_data.npy', X_test)
            np.save('./results/test_labels.npy', y_test)
            np.save('./results/val_data.npy', X_val)
            np.save('./results/val_labels.npy', y_val)
            print("原始測試數據和驗證數據已保存到 ./results/")
            all_text_lines.append("原始測試數據和驗證數據已保存到 ./results/")
        
        # 創建數據加載器
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)
        
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        results = {}
        
        # 評估驗證集
        if evaluate_val:
            print("\n" + "="*60)
            print("評估驗證集")
            print("="*60)
            all_text_lines.append("\n" + "="*60)
            all_text_lines.append("評估驗證集")
            all_text_lines.append("="*60)
            
            y_true_val, y_pred_val, y_proba_val = self.evaluate_dataset(val_loader, "Validation")
            result_val, text_val = self.print_evaluation_metrics(y_true_val, y_pred_val, y_proba_val, "Validation", return_text=True)
            results['val'] = result_val
            all_text_lines.append(text_val)
            
            # 保存結果
            np.save('./results/val_predictions.npy', y_pred_val)
            np.save('./results/val_probabilities.npy', y_proba_val)
            np.save('./results/val_true_labels.npy', y_true_val)
            print("\n驗證集預測結果已保存到 ./results/")
            all_text_lines.append("\n驗證集預測結果已保存到 ./results/")
        
        # 評估測試集
        if evaluate_test:
            print("\n" + "="*60)
            print("評估測試集")
            print("="*60)
            all_text_lines.append("\n" + "="*60)
            all_text_lines.append("評估測試集")
            all_text_lines.append("="*60)
            
            y_true_test, y_pred_test, y_proba_test = self.evaluate_dataset(test_loader, "Test")
            result_test, text_test = self.print_evaluation_metrics(y_true_test, y_pred_test, y_proba_test, "Test", return_text=True)
            results['test'] = result_test
            all_text_lines.append(text_test)
            
            # 保存結果
            np.save('./results/test_predictions.npy', y_pred_test)
            np.save('./results/test_probabilities.npy', y_proba_test)
            np.save('./results/test_true_labels.npy', y_true_test)
            print("\n測試集預測結果已保存到 ./results/")
            all_text_lines.append("\n測試集預測結果已保存到 ./results/")
        
        # 過擬合分析（如果同時評估驗證集和測試集）
        if evaluate_val and evaluate_test:
            print("\n" + "="*60)
            print("過擬合分析")
            print("="*60)
            all_text_lines.append("\n" + "="*60)
            all_text_lines.append("過擬合分析")
            all_text_lines.append("="*60)
            
            val_balanced_acc = results['val']['balanced_accuracy']
            test_balanced_acc = results['test']['balanced_accuracy']
            overfitting_gap = val_balanced_acc - test_balanced_acc
            
            overfitting_text = []
            overfitting_text.append(f"驗證集 Balanced Accuracy: {val_balanced_acc:.4f}")
            overfitting_text.append(f"測試集 Balanced Accuracy: {test_balanced_acc:.4f}")
            overfitting_text.append(f"過擬合差距: {overfitting_gap:.4f}")
            
            print(f"驗證集 Balanced Accuracy: {val_balanced_acc:.4f}")
            print(f"測試集 Balanced Accuracy: {test_balanced_acc:.4f}")
            print(f"過擬合差距: {overfitting_gap:.4f}")
            
            if overfitting_gap > 0.15:
                warning = "⚠️ 警告：存在明顯過擬合（差距 > 0.15）"
                overfitting_text.append(warning)
                print(warning)
            elif overfitting_gap > 0.10:
                warning = "⚠️ 注意：存在一定過擬合（差距 > 0.10）"
                overfitting_text.append(warning)
                print(warning)
            else:
                warning = "✓ 過擬合程度在可接受範圍內"
                overfitting_text.append(warning)
                print(warning)
            
            overfitting_text.append("="*60)
            all_text_lines.append('\n'.join(overfitting_text))
            print("="*60)
        
        # 添加圖表保存信息
        all_text_lines.append("\n" + "="*60)
        all_text_lines.append("生成的可視化圖表")
        all_text_lines.append("="*60)
        if evaluate_val:
            all_text_lines.append("驗證集圖表:")
            all_text_lines.append("  - confusion_matrix_validation.png (混淆矩陣)")
            all_text_lines.append("  - roc_curves_validation.png (ROC 曲線)")
            all_text_lines.append("  - class_distribution_validation.png (類別分布)")
        if evaluate_test:
            all_text_lines.append("測試集圖表:")
            all_text_lines.append("  - confusion_matrix_test.png (混淆矩陣)")
            all_text_lines.append("  - roc_curves_test.png (ROC 曲線)")
            all_text_lines.append("  - class_distribution_test.png (類別分布)")
        all_text_lines.append("="*60)
        
        # 保存所有結果到 txt 文件
        if save_to_file:
            output_file = './results/evaluation_report.txt'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(all_text_lines))
            print(f"\n完整評估報告已保存到: {output_file}")
            print("\n生成的可視化圖表:")
            if evaluate_val:
                print("  驗證集: confusion_matrix_validation.png, roc_curves_validation.png, class_distribution_validation.png")
            if evaluate_test:
                print("  測試集: confusion_matrix_test.png, roc_curves_test.png, class_distribution_test.png")
        
        return results


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='評估訓練好的 Conformer 模型')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型文件路徑（如果為 None 則使用 best_model.pth）')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='檢查點文件路徑（優先於 model_path）')
    parser.add_argument('--base_dir', type=str, default='../',
                        help='數據根目錄')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='檢查點目錄')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--evaluate_val', action='store_true', default=True,
                        help='評估驗證集')
    parser.add_argument('--evaluate_test', action='store_true', default=True,
                        help='評估測試集')
    parser.add_argument('--no_val', action='store_true',
                        help='不評估驗證集')
    parser.add_argument('--no_test', action='store_true',
                        help='不評估測試集')
    
    args = parser.parse_args()
    
    # 創建評估器
    evaluator = ConformerEvaluator(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        base_dir=args.base_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        emb_size=40,
        depth=6,
        n_classes=3,
        window_length=2.0,
        target_sfreq=500,
        n_channels=19,
        target_timepoints=1000,
    )
    
    # 執行評估
    evaluate_val = args.evaluate_val and not args.no_val
    evaluate_test = args.evaluate_test and not args.no_test
    
    results = evaluator.evaluate(
        evaluate_val=evaluate_val,
        evaluate_test=evaluate_test
    )
    
    print("\n" + "="*60)
    print("評估完成")
    print("="*60)


if __name__ == "__main__":
    main()

