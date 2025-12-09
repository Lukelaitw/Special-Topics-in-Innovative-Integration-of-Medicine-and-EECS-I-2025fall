import pandas as pd
import numpy as np
import mne
import os
from pygsp import graphs, utils
from scipy.spatial import distance_matrix
from scipy.stats import entropy
import networkx as nx
from sklearn.cluster import spectral_clustering

def compute_total_variation(W, data_values):
    """
    向量化版本的总变分计算，比双重循环快得多
    
    总变分公式: TV = sqrt(sum_{i,j} w_{ij} * ||x_j - x_i||^2)
    """
    # 使用广播计算所有 (i,j) 对的差值
    # data_values[i] 形状: (D,), data_values[j] 形状: (D,)
    # 我们需要计算所有 i,j 对的差值
    N = data_values.shape[0]
    
    # 方法1: 使用广播 (更高效)
    # 扩展维度: data_values[i] -> (N, 1, D), data_values[j] -> (1, N, D)
    data_i = data_values[:, np.newaxis, :]  # (N, 1, D)
    data_j = data_values[np.newaxis, :, :]  # (1, N, D)
    
    # 计算所有差值: (N, N, D)
    differences = data_j - data_i
    
    # 计算每个差值的 L2 范数的平方: (N, N)
    squared_norms = np.sum(differences ** 2, axis=2)
    
    # 与权重矩阵相乘并求和
    TV = np.sum(W * squared_norms)
    
    return np.sqrt(TV)

dir_path = r'/home/b12901075/eecsmed/ds004504/derivatives'
file_list = [
    os.path.join(root, file) 
    for root, dirs, files in os.walk(dir_path) 
    for file in files 
    if file.endswith(".set")
]

n_files = len(file_list)
if n_files == 0:
    raise ValueError(f"No .set files found in directory {dir_path}.")

print(f'Found {n_files} .set files.')

channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

data_list = []
features = {}

for i, file in enumerate(file_list):
    print(f'Processing {file} ({i+1}/{n_files})')
    raw = mne.io.read_raw_eeglab(file)
    data = raw.get_data(picks=channel_names)
    transposed_data = np.transpose(data)
    data = pd.DataFrame(transposed_data, columns=channel_names)
    data = data.groupby(data.index // 50).median()
    data_list.append(data)

    # GSP analysis
    distances = distance_matrix(data.values, data.values)
    theta, k = 1.0, 1.0 
    W = np.exp(-distances**2 / theta**2)
    W[distances > k] = 0
    np.fill_diagonal(W, 0)
    G = graphs.Graph(W)
    L = G.L.toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    X_GdataT = eigenvectors.T @ data.values
    C = np.cov(X_GdataT)
    T = eigenvectors.T.conj() @ C @ eigenvectors
    r = np.linalg.norm(np.diag(T)) / np.linalg.norm(T, 'fro')
    P = L @ data.values
    Y = np.sum(data.values * P)**2
    TV = compute_total_variation(W, data.values)

    # Spectral Graph Features
    graph_energy = np.sum(np.abs(eigenvalues))
    # 计算 spectral entropy：使用特征值的归一化分布作为概率分布
    # 将特征值转换为非负值并归一化
    eigenvals_abs = np.abs(eigenvalues)
    eigenvals_normalized = eigenvals_abs / (np.sum(eigenvals_abs) + 1e-10)  # 添加小值避免除零
    spectral_entropy = entropy(eigenvals_normalized)

    # Graph Signal Features
    signal_energy = np.sum(np.square(data.values))
    signal_power = np.var(data.values)

    # Graph Modularity and Community Structure
    labels = spectral_clustering(W)
    unique_labels = len(np.unique(labels))

    # Graph Degree Distribution
    degree_distribution = np.sum(W, axis=0)

    # Graph Diffusion Characteristics
    heat_trace = np.trace(np.exp(-L))
    diffusion_distance = np.sum(np.exp(-L))

    # Aggregating Features
    features[os.path.basename(file)] = {
        'stationary_ratio': r, 
        'Tik-norm': Y, 
        'Total_Variation': TV,
        'graph_energy': graph_energy,
        'spectral_entropy': spectral_entropy,
        'signal_energy': signal_energy,
        'signal_power': signal_power,
        'unique_clusters': unique_labels,
        'avg_degree': np.mean(degree_distribution),
        'heat_trace': heat_trace,
        'diffusion_distance': diffusion_distance
    }

features_data = pd.DataFrame(features).T
features_data.to_csv('features_tv.csv', index_label='participant_id')