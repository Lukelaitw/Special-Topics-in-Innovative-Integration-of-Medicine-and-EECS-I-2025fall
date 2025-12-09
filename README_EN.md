# Special Topics in Innovative Integration of Medicine and EECS I (2025 Fall)

**English** | [繁體中文](README.md)

---

This project is for the course "Special Topics in Innovative Integration of Medicine and EECS I (2025 Fall)", focusing on neural system disease classification using electroencephalography (EEG) signals. The project includes two different approaches: a deep learning method (EEG-Conformer) and a traditional machine learning method (EEG-Hierarchical-baseline).

## 🎯 Project Overview

This project aims to develop and compare different machine learning methods for automatically identifying neural system diseases from EEG signals, specifically:
- **Alzheimer's Disease (AD)**
- **Frontotemporal Dementia (FTD)**
- **Control Group (CN)**

The project uses the **ds004504** dataset, containing EEG recordings from 88 subjects.

## 📁 Project Structure

```
Special-Topics-in-Innovative-Integration-of-Medicine-and-EECS-I-2025fall/
├── README.md                          # This file (Traditional Chinese)
├── README_EN.md                       # English version
├── EEG-Conformer/                     # Deep learning approach: Conformer model
│   ├── conformer.py                   # Core Conformer model implementation
│   ├── conformer_train_with_svm_data_split.py  # Training script
│   ├── evaluate_conformer_model.py    # Model evaluation script
│   ├── visualization/                 # Visualization tools
│   ├── checkpoints-1/                 # Trained model weights
│   ├── results/                       # Evaluation results and visualizations
│   └── README.md                      # Conformer project documentation
├── EEG-Hierarchical-baseline/         # Traditional ML approach: Hierarchical classification
│   ├── baseline.ipynb                 # Baseline classifier (single-stage)
│   ├── two-stage-classification.ipynb # Two-stage classifier
│   ├── gsp_feature_extraction.py      # Graph Signal Processing feature extraction
│   └── features_tv.csv                # Extracted feature data
└── asset/                             # Experimental results and resources
    ├── baseline_results/              # Baseline method results
    │   ├── logistic_regression/       # Logistic regression results
    │   ├── random_forest/             # Random forest results
    │   └── SVM/                       # Support Vector Machine results
    └── two_stage_results/             # Two-stage method results
        ├── logistic_regression/       # Logistic regression results
        ├── random_forest/             # Random forest results
        └── SVM/                       # Support Vector Machine results
```

## 🔬 Method Overview

### 1. EEG-Conformer (Deep Learning Approach)

EEG-Conformer is a hybrid architecture combining Convolutional Neural Networks (CNN) and Transformer, specifically designed for EEG signal processing.

**Key Features:**
- End-to-end training without manual feature engineering
- Combines CNN's local feature extraction with Transformer's global dependency modeling
- Provides visualization tools such as Class Activation Topography (CAT)
- Uses SVM data splitting strategy (30 training, 31 validation, 27 test samples)

**Detailed Documentation:** See [EEG-Conformer/README.md](EEG-Conformer/README.md)

### 2. EEG-Hierarchical-baseline (Traditional Machine Learning Approach)

The hierarchical classification method uses traditional machine learning classifiers combined with Graph Signal Processing (GSP) feature extraction.

**Key Features:**
- **Single-stage Classification (Baseline)**: Direct three-class classification (AD/FTD/CN)
- **Two-stage Classification**: First distinguish disease group (AD+FTD) from control group (CN), then distinguish AD from FTD
- Uses Graph Signal Processing to extract features such as Total Variation
- Supports multiple classifiers: Logistic Regression, Random Forest, SVM

**Detailed Documentation:** See the respective Jupyter Notebook files

## 📊 Dataset

This project uses the **ds004504** dataset:
- **Total Subjects**: 88
- **Class Distribution**: AD, FTD, CN
- **Data Format**: BIDS format EEG data (.set files)
- **Data Split**:
  - Training set: 30 samples
  - Validation set: 31 samples
  - Test set: 27 samples

**Important Note:** When using this project or dataset, please cite the relevant papers (see Citation section).

## 🚀 Quick Start

### Environment Setup

#### 1. Install Python Dependencies

```bash
# Install PyTorch (choose according to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy pandas mne scipy scikit-learn einops matplotlib seaborn pygsp networkx
```

#### 2. Prepare Data

Ensure the dataset path is correctly configured. Modify the data path in relevant scripts:

```python
BASE_DIR = '/path/to/your/data'
DATASET_DIR = os.path.join(BASE_DIR, 'ds004504')
```

### Running EEG-Conformer

```bash
cd EEG-Conformer
python conformer_train_with_svm_data_split.py
```

### Running EEG-Hierarchical-baseline

```bash
cd EEG-Hierarchical-baseline
# Open and run using Jupyter Notebook
jupyter notebook baseline.ipynb
# or
jupyter notebook two-stage-classification.ipynb
```

## 📈 Experimental Results

Experimental results are saved in the `asset/` directory:

- **baseline_results/**: Single-stage classification results
  - Includes confusion matrices, evaluation metrics, etc.
- **two_stage_results/**: Two-stage classification results
  - Includes confusion matrices, evaluation metrics, prediction CSV files, etc.

### Result File Description

Each result directory contains:
- `confusion_matrix_test.png`: Test set confusion matrix
- `confusion_matrix_validation.png`: Validation set confusion matrix
- `evaluation_metrics.txt`: Evaluation metrics (accuracy, precision, recall, F1-score, etc.)
- `test_predictions.csv` / `validation_predictions.csv`: Prediction results (two-stage method only)

## 🔧 Main Dependencies

- **PyTorch**: Deep learning framework
- **MNE**: EEG data processing
- **scikit-learn**: Traditional machine learning methods
- **PyGSP**: Graph Signal Processing
- **NumPy, Pandas**: Data processing
- **Matplotlib, Seaborn**: Visualization

## 📝 Citation

If you use this project or related datasets, please cite:

### Citing Original Paper

This project is based on the following paper:

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

### Citing Dataset

This project uses the ds004504 dataset. Please cite the following dataset paper:

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

### Citing Related Resources

```bibtex
@misc{Xmootoo2025,
  author = {Xmootoo},
  title = {Applying the Graph Discrete Fourier Transform to EEG Data for Alzheimer Disease Detection},
  howpublished = {\url{https://github.com/xmootoo/gsp-alzheimer-detection}},
  note = {Accessed November 24, 2025},
  year = {2025}
}

@misc{OpenNeuro,
  title = {OpenNeuro},
  howpublished = {\url{https://openneuro.org/}},
  note = {Accessed November 30, 2025},
  year = {2025}
}
```

## 👥 Authors

[To be added: Author information]

## 📄 License

[To be added: License information]

## 🔗 Related Resources

- [EEG-Conformer Detailed Documentation](EEG-Conformer/README.md)
- [EEG-Conformer Visualization Tools](EEG-Conformer/visualization/README_visualization.md)
- [ds004504 Dataset](https://openneuro.org/datasets/ds004504)

## 📧 Contact

For questions or suggestions, please contact: [To be added]

---

**Last Updated:** December 2025
