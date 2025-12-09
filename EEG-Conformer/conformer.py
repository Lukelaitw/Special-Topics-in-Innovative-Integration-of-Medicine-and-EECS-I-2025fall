"""
EEG Conformer 

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

import argparse
import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import datetime
import sys
import scipy.io

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd
from torchvision.models import vgg19

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from common_spatial_pattern import csp

import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('./TensorBoardX/')


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (19, 1), (1, 1)),  # 調整為 19 通道以適配 ds004504
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3)  # 調整為 3 類別以適配 ds004504 (AD/FTD/CN)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=3, **kwargs):  # 調整為 3 類別以適配 ds004504
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub=None, load_all_subjects=True, n_total_subjects=88, checkpoint_dir='./checkpoints', resume_from=None):
        super(ExP, self).__init__()
        self.batch_size = 4
        self.n_epochs = 2000
        self.c_dim = 3  # 調整為 3 類別以適配 ds004504
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub  # 如果 load_all_subjects=True，這個參數不會使用
        self.load_all_subjects = load_all_subjects  # 是否混合所有受試者的資料
        self.n_total_subjects = n_total_subjects  # ds004504 總共有 88 個受試者

        self.start_epoch = 0
        self.root = '../CTNet/mymat_ds004504/'  # 調整為 ds004504 資料路徑
        
        # Checkpoint 相關設定
        self.checkpoint_dir = checkpoint_dir
        self.resume_from = resume_from
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if load_all_subjects:
            self.log_write = open("./results/log_all_subjects.txt", "a")  # 改為 append 模式以支援 resume
        else:
            self.log_write = open("./results/log_subject%d.txt" % self.nSub, "a")  # 改為 append 模式以支援 resume


        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = self.model.cuda()
        # summary(self.model, (1, 19, 1000))  # 調整為 19 通道以適配 ds004504

    def save_checkpoint(self, epoch, optimizer, bestAcc, averAcc, num, filename=None):
        """保存 checkpoint"""
        if filename is None:
            if self.load_all_subjects:
                filename = os.path.join(self.checkpoint_dir, 'checkpoint_all_subjects.pth')
            else:
                filename = os.path.join(self.checkpoint_dir, f'checkpoint_subject_{self.nSub}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # DataParallel 需要 .module
            'optimizer_state_dict': optimizer.state_dict(),
            'bestAcc': bestAcc,
            'averAcc': averAcc,
            'num': num,
            'lr': self.lr,
            'batch_size': self.batch_size,
        }
        torch.save(checkpoint, filename)
        print(f'Checkpoint 已保存: {filename} (Epoch {epoch})')
    
    def load_checkpoint(self, optimizer=None):
        """載入 checkpoint"""
        if self.resume_from is None:
            return False
        
        if not os.path.exists(self.resume_from):
            print(f'警告: Checkpoint 文件不存在: {self.resume_from}')
            return False
        
        print(f'正在載入 checkpoint: {self.resume_from}')
        checkpoint = torch.load(self.resume_from)
        
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        bestAcc = checkpoint.get('bestAcc', 0)
        averAcc = checkpoint.get('averAcc', 0)
        num = checkpoint.get('num', 0)
        
        # 驗證載入的值是否合理
        # averAcc 應該是累加值，所以如果 num > 0，averAcc 應該 <= num（理論上）
        # 但如果 averAcc 看起來像是平均值（< 1），可能是舊版本的 checkpoint
        if num > 0 and averAcc > 0:
            # 如果 averAcc 看起來像是平均值（小於等於 1），可能是舊版本的 checkpoint
            # 需要將其轉換回累加值
            if averAcc <= 1.0 and averAcc > 0:
                print(f'警告: 檢測到 averAcc ({averAcc:.6f}) 可能是平均值而非累加值，正在轉換...')
                averAcc = averAcc * num
                print(f'  已轉換為累加值: {averAcc:.6f} (num={num})')
        
        print(f'Checkpoint 載入成功: 從 Epoch {self.start_epoch} 繼續訓練')
        # 計算並顯示平均準確率（用於顯示）
        if num > 0:
            avg_acc_display = averAcc / num
            print(f'  最佳準確率: {bestAcc:.6f}, 累加準確率: {averAcc:.6f}, 平均準確率: {avg_acc_display:.6f}, num: {num}')
        else:
            print(f'  最佳準確率: {bestAcc:.6f}, 累加準確率: {averAcc:.6f}, num: {num}')
        
        return True, bestAcc, averAcc, num

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        for cls4aug in range(3):  # 調整為 3 類別以適配 ds004504
            cls_idx = np.where(label == cls4aug)  # ds004504 標籤為 0, 1, 2，不需要 +1
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 3), 1, 19, 1000))  # 調整為 19 通道
            for ri in range(int(self.batch_size / 3)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 3)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label).cuda()  # ds004504 標籤已經是 0, 1, 2，不需要 -1
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):
        # ! please please recheck if you need validation set 
        # ! and the data segement compared methods used

        if self.load_all_subjects:
            # 跨受試者訓練：載入所有受試者的資料並混合
            print(f'開始載入所有 {self.n_total_subjects} 個受試者的資料...')
            all_train_data_list = []
            all_train_label_list = []
            all_test_data_list = []
            all_test_label_list = []
            
            for sub_id in range(1, self.n_total_subjects + 1):
                try:
                    # 載入訓練資料
                    train_file = self.root + 'D%03dT.mat' % sub_id
                    if not os.path.exists(train_file):
                        print(f'警告: 找不到文件 {train_file}，跳過受試者 {sub_id}')
                        continue
                    
                    train_data_mat = scipy.io.loadmat(train_file)
                    train_data = train_data_mat['data']  # (trials, 19, 1000)
                    train_label = train_data_mat['label'].flatten()  # (trials,)
                    
                    # 載入測試資料
                    test_file = self.root + 'D%03dE.mat' % sub_id
                    if not os.path.exists(test_file):
                        print(f'警告: 找不到文件 {test_file}，跳過受試者 {sub_id}')
                        continue
                    
                    test_data_mat = scipy.io.loadmat(test_file)
                    test_data = test_data_mat['data']  # (trials, 19, 1000)
                    test_label = test_data_mat['label'].flatten()  # (trials,)
                    
                    all_train_data_list.append(train_data)
                    all_train_label_list.append(train_label)
                    all_test_data_list.append(test_data)
                    all_test_label_list.append(test_label)
                    
                    if sub_id % 10 == 0:
                        print(f'  已載入 {sub_id}/{self.n_total_subjects} 個受試者')
                        
                except Exception as e:
                    print(f'警告: 無法載入受試者 {sub_id} 的資料，跳過: {e}')
                    continue
            
            print(f'合併所有受試者的資料...')
            # 合併所有受試者的資料
            self.train_data = np.concatenate(all_train_data_list, axis=0)  # (total_trials, 19, 1000)
            self.train_label = np.concatenate(all_train_label_list, axis=0)  # (total_trials,)
            self.test_data = np.concatenate(all_test_data_list, axis=0)  # (total_trials, 19, 1000)
            self.test_label = np.concatenate(all_test_label_list, axis=0)  # (total_trials,)
            
            print(f'合併完成: 訓練集 {self.train_data.shape[0]} 個試次, 測試集 {self.test_data.shape[0]} 個試次')
            print(f'訓練集標籤分布: {np.bincount(self.train_label.astype(int))}')
            print(f'測試集標籤分布: {np.bincount(self.test_label.astype(int))}')
        else:
            # 單個受試者模式（保留原有功能以備不時之需）
            # train data - 調整為 ds004504 格式: D{subject_id:03d}T.mat
            self.total_data = scipy.io.loadmat(self.root + 'D%03dT.mat' % self.nSub)
            self.train_data = self.total_data['data']
            self.train_label = self.total_data['label'].flatten()

            # test data - 調整為 ds004504 格式: D{subject_id:03d}E.mat
            self.test_tmp = scipy.io.loadmat(self.root + 'D%03dE.mat' % self.nSub)
            self.test_data = self.test_tmp['data']
            self.test_label = self.test_tmp['label'].flatten()

        # ds004504 數據格式: (trials, channels, time_samples) = (trials, 19, 1000)
        # 需要轉換為: (trials, 1, channels, time_samples) = (trials, 1, 19, 1000)
        self.train_data = np.expand_dims(self.train_data, axis=1)  # 添加 conv channel 維度: (trials, 1, 19, 1000)
        self.test_data = np.expand_dims(self.test_data, axis=1)  # 添加 conv channel 維度: (trials, 1, 19, 1000)

        self.allData = self.train_data
        self.allLabel = self.train_label

        # 打亂訓練資料（混合所有受試者的試次）
        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        self.testData = self.test_data
        self.testLabel = self.test_label

        # 確保標籤在 [0, 1, 2] 範圍內
        unique_train_labels = np.unique(self.allLabel)
        unique_test_labels = np.unique(self.testLabel)
        print(f'訓練集標籤範圍: {unique_train_labels}, 最小值: {np.min(self.allLabel)}, 最大值: {np.max(self.allLabel)}')
        print(f'測試集標籤範圍: {unique_test_labels}, 最小值: {np.min(self.testLabel)}, 最大值: {np.max(self.testLabel)}')
        
        if np.min(self.allLabel) > 0:
            self.allLabel = self.allLabel - 1
            print(f'已將訓練標籤減 1，新範圍: {np.unique(self.allLabel)}')
        if np.min(self.testLabel) > 0:
            self.testLabel = self.testLabel - 1
            print(f'已將測試標籤減 1，新範圍: {np.unique(self.testLabel)}')

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel


    def train(self):

        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)  # ds004504 標籤已經是 0, 1, 2，不需要 -1

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label)  # ds004504 標籤已經是 0, 1, 2，不需要 -1
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # 嘗試載入 checkpoint
        if self.resume_from is not None:
            result = self.load_checkpoint(self.optimizer)
            if result and isinstance(result, tuple):
                _, bestAcc, averAcc, num = result
                print(f'從 checkpoint 恢復: bestAcc={bestAcc:.6f}, averAcc={averAcc:.6f}, num={num}')

        # Train the cnn model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        print(f'\n開始訓練，總共 {self.n_epochs} 個 epochs，每個 epoch 有 {total_step} 個批次')
        if self.start_epoch > 0:
            print(f'從 Epoch {self.start_epoch} 繼續訓練')
        print('=' * 60)
        
        # 清理初始 GPU 快取
        torch.cuda.empty_cache()
        
        for e in range(self.start_epoch, self.n_epochs):
            # in_epoch = time.time()
            self.model.train()
            epoch_start_time = time.time()
            last_outputs = None
            last_label = None
            last_loss = None
            
            for i, (img, label) in enumerate(self.dataloader):

                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                # 清理 augmentation 後的臨時變數記憶體
                del aug_data, aug_label
                torch.cuda.empty_cache()

                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label) 

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 保存最後一個批次的輸出用於計算訓練準確率
                if i == len(self.dataloader) - 1:
                    last_outputs = outputs.detach().clone()
                    last_label = label.detach().clone()
                    last_loss = loss.detach().clone()
                
                # 每 10 個批次顯示一次進度（避免輸出過多）
                if (i + 1) % 10 == 0 or (i + 1) == total_step:
                    print(f'  Epoch {e+1}/{self.n_epochs}, 批次 {i+1}/{total_step}, 當前損失: {loss.item():.6f}', flush=True)
                
                # 定期清理 GPU 快取（每 50 個批次清理一次，更頻繁地釋放記憶體）
                if (i + 1) % 50 == 0:
                    torch.cuda.empty_cache()
                
                # 釋放當前批次的變數（除了最後一個批次，需要保留用於計算訓練準確率）
                if i != len(self.dataloader) - 1:
                    # 安全刪除變數，避免重複刪除已刪除的變數
                    try:
                        del tok, outputs, loss
                    except NameError:
                        # 變數可能已經被刪除，忽略錯誤
                        pass
                    torch.cuda.empty_cache()


            # out_epoch = time.time()
            
            # Epoch 結束後清理記憶體
            torch.cuda.empty_cache()

            # test process
            if (e + 1) % 1 == 0:
                # 清理 GPU 快取
                torch.cuda.empty_cache()
                
                self.model.eval()
                with torch.no_grad():  # 測試時不需要計算梯度，節省記憶體
                    Tok, Cls = self.model(test_data)
                    loss_test = self.criterion_cls(Cls, test_label)
                    y_pred = torch.max(Cls, 1)[1]
                    acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                
                # 釋放測試相關的變數
                del Tok, Cls, loss_test
                torch.cuda.empty_cache()
                
                # 計算訓練準確率（使用最後一個批次的輸出）
                if last_outputs is not None and last_label is not None:
                    train_pred = torch.max(last_outputs, 1)[1]
                    train_acc = float((train_pred == last_label).cpu().numpy().astype(int).sum()) / float(last_label.size(0))
                    train_loss = last_loss.item() if last_loss is not None else 0.0
                else:
                    train_acc = 0.0
                    train_loss = 0.0

                epoch_time = time.time() - epoch_start_time
                print('Epoch:', e+1,
                      '  Train loss: %.6f' % train_loss,
                      '  Test accuracy is %.6f' % acc,
                      '  Train accuracy %.6f' % train_acc,
                      f'  耗時: {epoch_time:.2f}秒', flush=True)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                self.log_write.flush()  # 確保日誌立即寫入
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred
                
                # 釋放訓練相關的變數
                if last_outputs is not None:
                    del last_outputs, last_label, last_loss, train_pred
                    torch.cuda.empty_cache()
                
                # 定期保存 checkpoint（每 10 個 epoch 或最佳準確率更新時）
                if (e + 1) % 10 == 0 or acc > bestAcc:
                    self.save_checkpoint(e, self.optimizer, bestAcc, averAcc, num)
                    torch.cuda.empty_cache()


        # 保存最終模型和 checkpoint
        torch.save(self.model.module.state_dict(), 'model.pth')
        self.save_checkpoint(self.n_epochs - 1, self.optimizer, bestAcc, averAcc, num, 
                           filename=os.path.join(self.checkpoint_dir, 'final_checkpoint.pth'))
        
        # 計算平均準確率，確保 num > 0 且 averAcc 是累加值
        if num > 0:
            averAcc = averAcc / num
        else:
            print('警告: num 為 0，無法計算平均準確率')
            averAcc = 0.0
        
        # 確保平均準確率在合理範圍內（0-1）
        if averAcc > 1.0:
            print(f'警告: 平均準確率異常 ({averAcc:.6f})，可能是 checkpoint 載入問題')
            # 如果異常，重新計算（這不應該發生，但作為安全措施）
            averAcc = min(averAcc, 1.0)
        
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        self.log_write.close()
        
        # 最終清理
        torch.cuda.empty_cache()

        return bestAcc, averAcc, Y_true, Y_pred
        # writer.close()


def find_latest_checkpoint(checkpoint_dir, load_all_subjects=True, nsub=None):
    """自動找到最新的 checkpoint 文件"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找所有 .pth 文件
    all_checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    
    if not all_checkpoints:
        return None
    
    # 根據模式過濾相關的 checkpoint 文件
    checkpoint_files = []
    
    if load_all_subjects:
        # 跨受試者模式：查找所有相關的 checkpoint（all_subjects 或 final）
        for cp in all_checkpoints:
            cp_name = os.path.basename(cp)
            if 'all_subjects' in cp_name or 'final' in cp_name or cp_name.startswith('checkpoint'):
                checkpoint_files.append((cp, os.path.getmtime(cp)))
    else:
        # 單個受試者模式：查找特定受試者的 checkpoint
        subject_pattern = f'subject_{nsub}'
        for cp in all_checkpoints:
            cp_name = os.path.basename(cp)
            if subject_pattern in cp_name:
                checkpoint_files.append((cp, os.path.getmtime(cp)))
    
    # 如果沒有找到匹配的文件，使用所有 checkpoint 文件
    if not checkpoint_files:
        checkpoint_files = [(cp, os.path.getmtime(cp)) for cp in all_checkpoints]
    
    if not checkpoint_files:
        return None
    
    # 返回最新的文件（按修改時間排序）
    latest_checkpoint = max(checkpoint_files, key=lambda x: x[1])[0]
    return latest_checkpoint


def main():
    # 配置參數
    LOAD_ALL_SUBJECTS = True  # True: 跨受試者訓練（混合所有受試者的資料）, False: 單個受試者訓練
    N_TOTAL_SUBJECTS = 88  # ds004504 的受試者總數
    
    # Checkpoint 設定
    AUTO_RESUME = True  # True: 自動找到最新的 checkpoint 並恢復訓練, False: 手動指定或從頭開始
    RESUME_FROM = None  # 如果 AUTO_RESUME=False，可以手動指定 checkpoint 路徑，例如: './checkpoints/checkpoint_all_subjects.pth'
    
    # 如果啟用自動恢復，自動找到最新的 checkpoint
    if AUTO_RESUME:
        checkpoint_dir = './checkpoints'
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir, load_all_subjects=LOAD_ALL_SUBJECTS)
        if latest_checkpoint:
            RESUME_FROM = latest_checkpoint
            print(f'自動找到最新的 checkpoint: {RESUME_FROM}')
        else:
            RESUME_FROM = None
            print('未找到 checkpoint，將從頭開始訓練')
    
    # 檢查 results 目錄是否存在，不存在則創建
    if not os.path.exists('./results'):
        os.makedirs('./results')
    result_write = open("./results/sub_result.txt", "w")
    
    if LOAD_ALL_SUBJECTS:
        # 跨受試者訓練模式：混合所有受試者的資料進行訓練
        print('=' * 60)
        if RESUME_FROM:
            print('跨受試者訓練模式：從 checkpoint 恢復訓練')
        else:
            print('跨受試者訓練模式：混合所有受試者的資料')
        print('=' * 60)
        
        starttime = datetime.datetime.now()
        
        # 只有在不恢復訓練時才設定新的隨機種子
        if RESUME_FROM is None:
            seed_n = np.random.randint(2021)
            print('seed is ' + str(seed_n))
            random.seed(seed_n)
            np.random.seed(seed_n)
            torch.manual_seed(seed_n)
            torch.cuda.manual_seed(seed_n)
            torch.cuda.manual_seed_all(seed_n)
        else:
            print(f'從 checkpoint 恢復訓練: {RESUME_FROM}')
            seed_n = None  # 恢復訓練時不設定新種子
        
        exp = ExP(load_all_subjects=True, n_total_subjects=N_TOTAL_SUBJECTS, resume_from=RESUME_FROM)
        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('跨受試者訓練模式\n')
        if seed_n is not None:
            result_write.write('Seed is: ' + str(seed_n) + "\n")
        if RESUME_FROM:
            result_write.write('Resumed from: ' + str(RESUME_FROM) + "\n")
        result_write.write('The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        
        endtime = datetime.datetime.now()
        print('訓練時間: ' + str(endtime - starttime))
        
    else:
        # 單個受試者訓練模式（保留原有功能）
        best = 0
        aver = 0
        
        for i in range(N_TOTAL_SUBJECTS):
            starttime = datetime.datetime.now()

            seed_n = np.random.randint(2021)
            print('seed is ' + str(seed_n))
            random.seed(seed_n)
            np.random.seed(seed_n)
            torch.manual_seed(seed_n)
            torch.cuda.manual_seed(seed_n)
            torch.cuda.manual_seed_all(seed_n)

            print('Subject %d' % (i+1))
            # 單個受試者模式也可以支援 resume
            subject_resume_from = None
            if AUTO_RESUME:
                latest_checkpoint = find_latest_checkpoint('./checkpoints', load_all_subjects=False, nsub=i+1)
                if latest_checkpoint:
                    subject_resume_from = latest_checkpoint
                    print(f'  自動找到受試者 {i+1} 的 checkpoint: {subject_resume_from}')
            exp = ExP(i + 1, load_all_subjects=False, resume_from=subject_resume_from)

            bestAcc, averAcc, Y_true, Y_pred = exp.train()
            print('THE BEST ACCURACY IS ' + str(bestAcc))
            result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
            result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
            result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

            endtime = datetime.datetime.now()
            print('subject %d duration: '%(i+1) + str(endtime - starttime))
            best = best + bestAcc
            aver = aver + averAcc
            if i == 0:
                yt = Y_true
                yp = Y_pred
            else:
                yt = torch.cat((yt, Y_true))
                yp = torch.cat((yp, Y_pred))

        best = best / N_TOTAL_SUBJECTS
        aver = aver / N_TOTAL_SUBJECTS

        result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
        result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    
    result_write.close()


if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
