"""
使用 SVM 資料分類流程訓練 Conformer 模型

此腳本參考 SVM 分類程式的資料處理流程：
- 使用相同的資料分割策略（訓練集 30, 驗證集 31, 測試集 27）
- 從原始 EEG 數據載入並處理
- 使用相同的標籤編碼方式
"""

import os
os.environ['WANDB_API_KEY'] = "YOUR_WANDB_API_KEY"
import sys
import numpy as np
import pandas as pd
import mne
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')

# 導入 Conformer 模型
# 確保可以從當前目錄或父目錄導入
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformer import Conformer

# GPU 設定
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import time
import datetime
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True


class EEGDataset(Dataset):
    """EEG 數據集類別"""
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ConformerTrainer:
    """使用 SVM 資料分割策略的 Conformer 訓練器"""
    
    def __init__(
        self,
        base_dir='/ibmnas/427/bachelors/b12901077/eeg',
        dataset_dir=None,
        checkpoint_dir='./checkpoints',
        batch_size=4,
        n_epochs=2000,
        lr=0.0002,
        emb_size=40,
        depth=6,
        n_classes=3,
        window_length=2.0,  # 秒
        target_sfreq=500,   # 目標採樣率
        n_channels=19,      # EEG 通道數
        target_timepoints=1000,  # 目標時間點數
        resume_from=None,    # 從檢查點恢復訓練的路徑
        gradient_accumulation_steps=4  # 梯度累積步數，有效批次大小 = batch_size * gradient_accumulation_steps
    ):
        self.base_dir = base_dir
        if dataset_dir is None:
            self.dataset_dir = os.path.join(base_dir, 'ds004504')
        else:
            self.dataset_dir = dataset_dir
        
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.emb_size = emb_size
        self.depth = depth
        self.n_classes = n_classes
        self.window_length = window_length
        self.target_sfreq = target_sfreq
        self.n_channels = n_channels
        self.target_timepoints = target_timepoints
        self.resume_from = resume_from
        self.start_epoch = 0  # 從哪個 epoch 開始訓練
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 創建檢查點目錄
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        # 創建結果目錄
        if not os.path.exists('./results'):
            os.makedirs('./results')
        
        # 設備設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        
        # 損失函數
        self.criterion_cls = nn.CrossEntropyLoss().to(self.device)
        
        # 模型
        self.model = Conformer(emb_size=emb_size, depth=depth, n_classes=n_classes)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.to(self.device)
        
        # 優化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        
        # 標籤編碼器
        self.le = LabelEncoder()
        
        print("="*60)
        print("Conformer 訓練器初始化完成")
        print("="*60)
        print(f"設備: {self.device}")
        print(f"批次大小: {self.batch_size}")
        print(f"梯度累積步數: {self.gradient_accumulation_steps}")
        print(f"有效批次大小: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"訓練輪數: {self.n_epochs}")
        print(f"學習率: {self.lr}")
        print(f"嵌入大小: {self.emb_size}")
        print(f"深度: {self.depth}")
        print(f"類別數: {self.n_classes}")
        print("="*60)
    
    def load_eeg_data_from_subject(self, subject_id):
        """
        從單個受試者載入 EEG 數據
        
        Parameters
        ----------
        subject_id : int
            受試者 ID (1-88)
        
        Returns
        -------
        data : np.ndarray
            EEG 數據，形狀為 (trials, channels, timepoints)
        label : str
            受試者的群組標籤 (A/F/C)
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
                print(f"警告: 受試者 {subject_id} 的通道數為 {len(picks)}，預期為 {self.n_channels}")
                # 如果通道數不同，選擇前 n_channels 個通道
                if len(picks) > self.n_channels:
                    picks = picks[:self.n_channels]
                else:
                    # 如果通道數不足，跳過此受試者
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
                    # 裁剪到目標時間點
                    data = data[:, :, :self.target_timepoints]
                else:
                    # 使用插值擴展
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
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_all_data(self):
        """
        載入所有受試者的數據，並使用與 SVM 相同的分割策略
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
        subject_ids_list = []
        
        for idx, row in participants_df.iterrows():
            subject_id_str = row['participant_id'].replace('sub-', '')
            subject_id = int(subject_id_str)
            
            print(f"載入受試者 {subject_id}...", end=' ')
            data, group = self.load_eeg_data_from_subject(subject_id)
            
            if data is not None and group is not None:
                # 為每個 trial 創建標籤
                n_trials = data.shape[0]
                labels = [group] * n_trials
                
                all_data_list.append(data)
                all_labels_list.extend(labels)
                subject_ids_list.extend([subject_id] * n_trials)
                print(f"✓ ({n_trials} 個 trials)")
            else:
                print("✗")
        
        if len(all_data_list) == 0:
            raise ValueError("錯誤: 沒有成功載入任何數據！")
        
        # 合併所有數據
        X = np.concatenate(all_data_list, axis=0)  # (total_trials, channels, timepoints)
        y = np.array(all_labels_list)
        
        # 清理臨時列表以釋放記憶體
        del all_data_list, all_labels_list, subject_ids_list
        import gc
        gc.collect()
        
        print(f"\n總樣本數: {len(X)}")
        print(f"數據形狀: {X.shape}")
        print(f"標籤分布: {pd.Series(y).value_counts().to_dict()}")
        
        # 標準化數據（對每個通道分別標準化）
        print("\n標準化數據...")
        X_normalized = np.zeros_like(X)
        for i in range(X.shape[1]):  # 對每個通道
            channel_data = X[:, i, :]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            if std_val > 0:
                X_normalized[:, i, :] = (channel_data - mean_val) / std_val
            else:
                X_normalized[:, i, :] = channel_data
        
        X = X_normalized
        del X_normalized  # 釋放臨時變數
        import gc
        gc.collect()  # 強制垃圾回收
        
        # 編碼標籤
        # 如果 LabelEncoder 已經有狀態（從檢查點載入），使用 transform；否則使用 fit_transform
        if hasattr(self.le, 'classes_') and len(self.le.classes_) > 0:
            print("\n使用已載入的 LabelEncoder 狀態進行標籤編碼...")
            y_encoded = self.le.transform(y)
        else:
            print("\n初始化 LabelEncoder 並編碼標籤...")
            y_encoded = self.le.fit_transform(y)
        
        print(f"\n類別編碼: {dict(zip(self.le.classes_, range(len(self.le.classes_))))}")
        print(f"標籤分布 (編碼後): {dict(zip(self.le.classes_, np.bincount(y_encoded)))}")
        
        # 數據分割（目標比例：訓練集 30, 驗證集 31, 測試集 27）
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
        
        # 清理不需要的變數
        del X, y, X_train_val, y_train_val
        import gc
        gc.collect()
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def interaug(self, timg, label, max_aug_samples=None):
        """
        數據增強：Segmentation and Reconstruction (S&R)
        參考原始 conformer.py 的實現
        
        Parameters
        ----------
        timg : np.ndarray
            訓練數據
        label : np.ndarray
            訓練標籤
        max_aug_samples : int, optional
            最大增強樣本數，如果為 None 則使用 batch_size // 2（減少記憶體使用）
        """
        aug_data = []
        aug_label = []
        segment_length = self.target_timepoints // 8  # 1000 // 8 = 125
        
        # 限制增強樣本數量以減少記憶體使用
        if max_aug_samples is None:
            max_aug_samples = max(1, self.batch_size // 2)  # 只生成一半的增強樣本
        
        samples_per_class = max(1, max_aug_samples // self.n_classes)
        
        for cls4aug in range(self.n_classes):
            cls_idx = np.where(label == cls4aug)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            
            if len(tmp_data) == 0:
                continue
            
            tmp_aug_data = np.zeros((samples_per_class, 1, self.n_channels, self.target_timepoints))
            for ri in range(samples_per_class):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    start_idx = rj * segment_length
                    end_idx = (rj + 1) * segment_length
                    tmp_aug_data[ri, :, :, start_idx:end_idx] = tmp_data[rand_idx[rj], :, :, start_idx:end_idx]
            
            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:samples_per_class])
        
        if len(aug_data) > 0:
            aug_data = np.concatenate(aug_data)
            aug_label = np.concatenate(aug_label)
            aug_shuffle = np.random.permutation(len(aug_data))
            aug_data = aug_data[aug_shuffle, :, :, :]
            aug_label = aug_label[aug_shuffle]
            
            # 限制總增強樣本數不超過 max_aug_samples
            if len(aug_data) > max_aug_samples:
                aug_data = aug_data[:max_aug_samples]
                aug_label = aug_label[:max_aug_samples]
            
            aug_data = torch.from_numpy(aug_data).to(self.device)
            aug_data = aug_data.float()
            aug_label = torch.from_numpy(aug_label).to(self.device)
            aug_label = aug_label.long()
        else:
            aug_data = torch.empty((0, 1, self.n_channels, self.target_timepoints)).to(self.device)
            aug_label = torch.empty((0,), dtype=torch.long).to(self.device)
        
        return aug_data, aug_label
    
    def save_checkpoint(self, epoch, optimizer, best_acc, aver_acc, num, filename=None):
        """
        保存檢查點
        
        Parameters
        ----------
        epoch : int
            當前 epoch
        optimizer : torch.optim.Optimizer
            優化器
        best_acc : float
            最佳準確率
        aver_acc : float
            累加準確率（用於計算平均值）
        num : int
            驗證次數
        filename : str, optional
            保存的文件名，如果為 None 則使用默認名稱
        """
        if filename is None:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        
        # 清理 GPU 快取以釋放記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 將優化器狀態移到 CPU 以節省 GPU 記憶體
        optimizer_state_dict = optimizer.state_dict()
        # 將優化器狀態中的張量移到 CPU
        cpu_optimizer_state = {}
        for key, value in optimizer_state_dict.items():
            if isinstance(value, dict):
                cpu_optimizer_state[key] = {}
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        cpu_optimizer_state[key][k] = v.cpu()
                    else:
                        cpu_optimizer_state[key][k] = v
            elif isinstance(value, torch.Tensor):
                cpu_optimizer_state[key] = value.cpu()
            else:
                cpu_optimizer_state[key] = value
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': cpu_optimizer_state,  # 使用 CPU 版本的優化器狀態
            'best_acc': best_acc,
            'aver_acc': aver_acc,
            'num': num,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'emb_size': self.emb_size,
            'depth': self.depth,
            'n_classes': self.n_classes,
            'label_encoder_classes': self.le.classes_ if hasattr(self.le, 'classes_') and len(self.le.classes_) > 0 else None,  # 保存 LabelEncoder 狀態
        }
        
        # 使用 CPU 保存 checkpoint 以節省 GPU 記憶體
        torch.save(checkpoint, filename)
        print(f'Checkpoint 已保存: {filename} (Epoch {epoch})')
        
        # 清理臨時變數
        del cpu_optimizer_state, checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_checkpoint(self, optimizer=None):
        """
        載入檢查點
        
        Parameters
        ----------
        optimizer : torch.optim.Optimizer, optional
            優化器，如果提供則會載入優化器狀態
        
        Returns
        -------
        bool
            是否成功載入
        tuple or None
            如果成功，返回 (best_acc, aver_acc, num)，否則返回 None
        """
        if self.resume_from is None:
            return False, None
        
        if not os.path.exists(self.resume_from):
            print(f'警告: Checkpoint 文件不存在: {self.resume_from}')
            return False, None
        
        print(f'正在載入 checkpoint: {self.resume_from}')
        # 使用 CPU 載入以節省 GPU 記憶體，然後手動移動到正確設備
        checkpoint = torch.load(self.resume_from, map_location='cpu')
        
        # 載入模型狀態
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 載入優化器狀態（如果優化器狀態在 CPU，需要移到 GPU）
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            # 將優化器狀態中的張量移到正確的設備
            if torch.cuda.is_available():
                gpu_optimizer_state = {}
                for key, value in optimizer_state_dict.items():
                    if isinstance(value, dict):
                        gpu_optimizer_state[key] = {}
                        for k, v in value.items():
                            if isinstance(v, torch.Tensor):
                                gpu_optimizer_state[key][k] = v.to(self.device)
                            else:
                                gpu_optimizer_state[key][k] = v
                    elif isinstance(value, torch.Tensor):
                        gpu_optimizer_state[key] = value.to(self.device)
                    else:
                        gpu_optimizer_state[key] = value
                optimizer.load_state_dict(gpu_optimizer_state)
                del gpu_optimizer_state  # 清理臨時變數
            else:
                optimizer.load_state_dict(optimizer_state_dict)
        
        # 載入訓練狀態
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0)
        aver_acc = checkpoint.get('aver_acc', 0)
        num = checkpoint.get('num', 0)
        
        # 載入 LabelEncoder 狀態（如果存在）
        if 'label_encoder_classes' in checkpoint and checkpoint['label_encoder_classes'] is not None:
            self.le.classes_ = checkpoint['label_encoder_classes']
            print(f'已載入 LabelEncoder 狀態: {self.le.classes_}')
        
        # 驗證載入的值是否合理
        if num > 0 and aver_acc > 0:
            # 如果 aver_acc 看起來像是平均值（小於等於 1），可能是舊版本的 checkpoint
            # 需要將其轉換回累加值
            if aver_acc <= 1.0 and aver_acc > 0:
                print(f'警告: 檢測到 aver_acc ({aver_acc:.6f}) 可能是平均值而非累加值，正在轉換...')
                aver_acc = aver_acc * num
                print(f'  已轉換為累加值: {aver_acc:.6f} (num={num})')
        
        print(f'Checkpoint 載入成功: 從 Epoch {self.start_epoch} 繼續訓練')
        # 計算並顯示平均準確率（用於顯示）
        if num > 0:
            avg_acc_display = aver_acc / num
            print(f'  最佳準確率: {best_acc:.6f}, 累加準確率: {aver_acc:.6f}, 平均準確率: {avg_acc_display:.6f}, num: {num}')
        else:
            print(f'  最佳準確率: {best_acc:.6f}, 累加準確率: {aver_acc:.6f}, num: {num}')
        
        return True, (best_acc, aver_acc, num)
    
    def find_latest_checkpoint(self):
        """
        自動找到最新的檢查點文件
        
        Returns
        -------
        str or None
            最新的檢查點文件路徑，如果沒有找到則返回 None
        """
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        import glob
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint*.pth'))
        
        if not checkpoint_files:
            return None
        
        # 按修改時間排序，返回最新的
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint
    
    def load_label_encoder_from_checkpoint(self):
        """
        從檢查點載入 LabelEncoder 狀態（在載入數據之前調用）
        
        Returns
        -------
        bool
            是否成功載入 LabelEncoder 狀態
        """
        if self.resume_from is None:
            return False
        
        if not os.path.exists(self.resume_from):
            return False
        
        try:
            checkpoint = torch.load(self.resume_from, map_location='cpu')  # 使用 CPU 載入以節省 GPU 記憶體
            
            # 載入 LabelEncoder 狀態（如果存在）
            if 'label_encoder_classes' in checkpoint and checkpoint['label_encoder_classes'] is not None:
                self.le.classes_ = checkpoint['label_encoder_classes']
                print(f'已從檢查點載入 LabelEncoder 狀態: {self.le.classes_}')
                return True
        except Exception as e:
            print(f'警告: 載入 LabelEncoder 狀態時發生錯誤: {e}')
        
        return False
    
    def train(self):
        """訓練模型"""
        # 如果 resume，先載入 LabelEncoder 狀態（在載入數據之前）
        if self.resume_from is not None:
            self.load_label_encoder_from_checkpoint()
        
        # 載入數據（會使用已設置的 LabelEncoder 狀態）
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_all_data()
        
        # 創建數據加載器
        train_dataset = EEGDataset(X_train, y_train)
        val_dataset = EEGDataset(X_val, y_val)
        test_dataset = EEGDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 不再將整個測試集載入 GPU，改為分批驗證以節省記憶體
        # 測試數據保留在 CPU，驗證時才分批移到 GPU
        
        # 清理原始 numpy 數組（如果不再需要）
        # 注意：X_train, X_val, X_test 仍被 DataLoader 使用，所以不能刪除
        # 但可以清理一些臨時變數
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 訓練記錄
        best_acc = 0
        aver_acc = 0
        num = 0
        Y_true = None
        Y_pred = None
        
        # 嘗試載入檢查點（模型和優化器狀態）
        if self.resume_from is not None:
            success, checkpoint_data = self.load_checkpoint(self.optimizer)
            if success and checkpoint_data is not None:
                best_acc, aver_acc, num = checkpoint_data
        
        # 日誌文件（如果 resume 則使用追加模式，否則使用寫入模式）
        if self.start_epoch > 0:
            log_file = open("./results/log_conformer_svm_split.txt", "a")
            log_file.write(f"\n--- Resume from checkpoint at Epoch {self.start_epoch} ---\n")
            log_file.flush()
        else:
            log_file = open("./results/log_conformer_svm_split.txt", "w")
        
        print("\n" + "="*60)
        if self.start_epoch > 0:
            print(f"從 Epoch {self.start_epoch} 繼續訓練")
        else:
            print("開始訓練")
        print("="*60)
        
        for epoch in range(self.start_epoch, self.n_epochs):
            self.model.train()
            epoch_start_time = time.time()
            last_outputs = None
            last_label = None
            last_loss = None
            accumulated_loss = 0.0  # 用於累積損失顯示
            loss_count = 0  # 用於計算平均損失
            
            # 初始化梯度累積計數器
            self.optimizer.zero_grad()
            
            for batch_idx, (img, label) in enumerate(train_loader):
                img = Variable(img.to(self.device).type(self.Tensor))
                label = Variable(label.to(self.device).type(self.LongTensor))
                
                # 數據增強（限制增強樣本數量以減少記憶體使用）
                # 只生成最多 batch_size // 2 個增強樣本，避免批次大小翻倍
                aug_data, aug_label = self.interaug(X_train, y_train, max_aug_samples=self.batch_size // 2)
                if len(aug_data) > 0:
                    img = torch.cat((img, aug_data))
                    label = torch.cat((label, aug_label))
                
                # 清理 augmentation 後的臨時變數記憶體
                del aug_data, aug_label
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 前向傳播
                tok, outputs = self.model(img)
                loss = self.criterion_cls(outputs, label)
                
                # 將損失除以累積步數，以便在累積後的平均損失正確
                loss_scaled = loss / self.gradient_accumulation_steps
                
                # 反向傳播（累積梯度）
                loss_scaled.backward()
                
                # 累積損失用於顯示（使用原始損失值）
                accumulated_loss += loss.item()
                loss_count += 1
                
                # 保存最後一個批次的輸出用於計算訓練準確率
                if batch_idx == len(train_loader) - 1:
                    last_outputs = outputs.detach().clone()
                    last_label = label.detach().clone()
                    last_loss = loss.detach().clone()  # 保存原始損失值
                
                # 每當達到累積步數或是最後一個批次時，更新參數
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    # 梯度裁剪（可選，防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # 更新參數
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # 清理 GPU 快取
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 每 10 個批次顯示一次進度（顯示累積的平均損失）
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                    if loss_count > 0:
                        avg_loss = accumulated_loss / loss_count
                        print(f'  Epoch {epoch+1}/{self.n_epochs}, 批次 {batch_idx+1}/{len(train_loader)}, 平均損失: {avg_loss:.6f}', flush=True)
                        accumulated_loss = 0.0  # 重置累積損失
                        loss_count = 0  # 重置計數器
                
                # 定期清理 GPU 快取（每 50 個批次清理一次）
                if (batch_idx + 1) % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 釋放當前批次的變數（除了最後一個批次，需要保留用於計算訓練準確率）
                if batch_idx != len(train_loader) - 1:
                    try:
                        del tok, outputs, loss, img, label
                    except NameError:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Epoch 結束後清理記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 驗證（分批處理以節省記憶體）
            if (epoch + 1) % 1 == 0:
                # 清理 GPU 快取
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model.eval()
                all_preds = []
                all_labels = []
                total_loss = 0.0
                
                with torch.no_grad():
                    # 分批處理測試集，避免一次性載入整個測試集到 GPU
                    for test_batch_idx, (test_img, test_label_batch) in enumerate(test_loader):
                        test_img = Variable(test_img.to(self.device).type(self.Tensor))
                        test_label_batch = Variable(test_label_batch.to(self.device).type(self.LongTensor))
                        
                        Tok, Cls = self.model(test_img)
                        loss_test = self.criterion_cls(Cls, test_label_batch)
                        y_pred_batch = torch.max(Cls, 1)[1]
                        
                        # 累積預測和標籤
                        all_preds.append(y_pred_batch.detach().cpu())
                        all_labels.append(test_label_batch.detach().cpu())
                        total_loss += loss_test.item()
                        
                        # 立即釋放當前批次的變數
                        del Tok, Cls, loss_test, test_img, test_label_batch, y_pred_batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # 合併所有批次的結果
                all_preds = torch.cat(all_preds)
                all_labels = torch.cat(all_labels)
                acc = float((all_preds == all_labels).numpy().astype(int).sum()) / float(all_labels.size(0))
                avg_loss = total_loss / len(test_loader)
                
                # 保存 y_pred 和 test_labels 用於後續處理（如果需要）
                y_pred = all_preds.clone()  # 創建副本
                test_labels_for_save = all_labels.clone()  # 保存測試標籤
                
                # 清理臨時變數
                del all_preds, all_labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 計算訓練準確率（使用最後一個批次的輸出）
                if last_outputs is not None and last_label is not None:
                    train_pred = torch.max(last_outputs, 1)[1]
                    train_acc = float((train_pred == last_label).cpu().numpy().astype(int).sum()) / float(last_label.size(0))
                    train_loss = last_loss.item() if last_loss is not None else 0.0
                else:
                    # 如果沒有保存最後一個批次，使用新的批次計算
                    self.model.train()
                    with torch.no_grad():
                        last_batch_data, last_batch_label = next(iter(train_loader))
                        last_batch_data = Variable(last_batch_data.to(self.device).type(self.Tensor))
                        last_batch_label = Variable(last_batch_label.to(self.device).type(self.LongTensor))
                        _, train_outputs = self.model(last_batch_data)
                        train_pred = torch.max(train_outputs, 1)[1]
                        train_acc = float((train_pred == last_batch_label).cpu().numpy().astype(int).sum()) / float(last_batch_label.size(0))
                        train_loss = last_loss.item() if last_loss is not None else 0.0
                        
                        # 清理臨時變數
                        del last_batch_data, last_batch_label, train_outputs, train_pred
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                epoch_time = time.time() - epoch_start_time
                print(f'Epoch: {epoch+1}, Train loss: {train_loss:.6f}, Train acc: {train_acc:.6f}, Test acc: {acc:.6f}, 耗時: {epoch_time:.2f}秒', flush=True)
                
                log_file.write(f"{epoch}    {acc}\n")
                log_file.flush()
                
                num += 1
                aver_acc += acc
                if acc > best_acc:
                    best_acc = acc
                    # y_pred 和 test_labels_for_save 已經在 CPU 上了（從分批驗證中獲得）
                    Y_pred = y_pred.clone() if isinstance(y_pred, torch.Tensor) else y_pred
                    Y_true = test_labels_for_save.clone() if isinstance(test_labels_for_save, torch.Tensor) else test_labels_for_save
                    
                    # 保存最佳模型
                    if hasattr(self.model, 'module'):
                        torch.save(self.model.module.state_dict(), 
                                 os.path.join(self.checkpoint_dir, 'best_model.pth'))
                    else:
                        torch.save(self.model.state_dict(), 
                                 os.path.join(self.checkpoint_dir, 'best_model.pth'))
                    
                    # 清理記憶體
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 定期保存檢查點（每 10 個 epoch 或最佳準確率更新時）
                if (epoch + 1) % 10 == 0 or acc > best_acc:
                    # 在保存 checkpoint 前清理記憶體
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.save_checkpoint(epoch, self.optimizer, best_acc, aver_acc, num)
                    # save_checkpoint 內部已經清理了記憶體，這裡再次清理以確保
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 釋放訓練相關的變數
                if last_outputs is not None:
                    del last_outputs, last_label, last_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # 計算平均準確率
        if num > 0:
            aver_acc = aver_acc / num
        
        print("\n" + "="*60)
        print("訓練完成")
        print("="*60)
        print(f'最佳準確率: {best_acc:.6f}')
        print(f'平均準確率: {aver_acc:.6f}')
        print("="*60)
        
        log_file.write(f'The best accuracy is: {best_acc}\n')
        log_file.write(f'The average accuracy is: {aver_acc}\n')
        log_file.close()
        
        # 保存最終模型和檢查點
        if hasattr(self.model, 'module'):
            torch.save(self.model.module.state_dict(), 
                      os.path.join(self.checkpoint_dir, 'final_model.pth'))
        else:
            torch.save(self.model.state_dict(), 
                      os.path.join(self.checkpoint_dir, 'final_model.pth'))
        
        # 保存最終檢查點
        self.save_checkpoint(self.n_epochs - 1, self.optimizer, best_acc, aver_acc, num,
                           filename=os.path.join(self.checkpoint_dir, 'final_checkpoint.pth'))
        
        # 最終清理記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return best_acc, aver_acc, Y_true, Y_pred


def main():
    """主函數"""
    print(time.asctime(time.localtime(time.time())))
    
    # 設定參數
    BASE_DIR = '../'
    DATASET_DIR = os.path.join(BASE_DIR, 'ds004504')
    
    # Resume 設定
    AUTO_RESUME = True  # True: 自動找到最新的檢查點並恢復訓練, False: 手動指定或從頭開始
    RESUME_FROM = None  # 如果 AUTO_RESUME=False，可以手動指定檢查點路徑，例如: './checkpoints/checkpoint.pth'
    
    # 如果啟用自動恢復，自動找到最新的檢查點
    if AUTO_RESUME:
        checkpoint_dir = './checkpoints'
        if os.path.exists(checkpoint_dir):
            import glob
            checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint*.pth'))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                RESUME_FROM = latest_checkpoint
                print(f'自動找到最新的檢查點: {RESUME_FROM}')
            else:
                RESUME_FROM = None
                print('未找到檢查點，將從頭開始訓練')
        else:
            RESUME_FROM = None
            print('檢查點目錄不存在，將從頭開始訓練')
    
    # 創建訓練器
    # 為了減少記憶體使用，使用較小的批次大小和較大的梯度累積步數
    trainer = ConformerTrainer(
        base_dir=BASE_DIR,
        dataset_dir=DATASET_DIR,
        checkpoint_dir='./checkpoints',
        batch_size=2,  # 減小批次大小以減少記憶體使用
        n_epochs=200,
        lr=0.0002,
        emb_size=40,
        depth=6,
        n_classes=3,
        window_length=2.0,
        target_sfreq=500,
        n_channels=19,
        target_timepoints=1000,
        resume_from=RESUME_FROM,  # 從檢查點恢復訓練
        gradient_accumulation_steps=8  # 梯度累積步數，有效批次大小 = 2 * 8 = 16
    )
    
    # 開始訓練
    best_acc, aver_acc, Y_true, Y_pred = trainer.train()
    
    print(f'\n最終結果:')
    print(f'最佳準確率: {best_acc:.6f}')
    print(f'平均準確率: {aver_acc:.6f}')
    
    print(time.asctime(time.localtime(time.time())))


if __name__ == "__main__":
    main()

