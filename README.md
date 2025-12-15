# 🎵 Differentiation of Music Genre from an Audio File Using Neural Networks

This repository contains the full implementation and research work for classifying music genres using **Convolutional Neural Networks (CNN)** and **K-Nearest Neighbors (KNN)** based on **MFCC audio features**.  
This project is based on the research titled *“Differentiation of Music Genre from an Audio File Using Neural Networks”*.

## 📄 Abstract
This project explores deep learning and machine learning approaches — specifically **CNN** and **KNN** — to differentiate music genres from audio files. Instead of relying on computationally expensive spectrogram images, the model uses **Mel-Frequency Cepstral Coefficients (MFCCs)**, which significantly reduce training time and computational cost. Experiments were conducted using the **GTZAN dataset** (1000 audio files across 10 genres), achieving strong performance with up to **98% training accuracy** using CNN and **73% test accuracy** using KNN.

## 📂 Project Structure
```
Music_Genre_Classification/
│
├── paper/
│   └── Springer_Differentiation_of_Music_Genre.pdf
│
├── notebooks/
│   └── research_paper_updated.ipynb
│
├── src/
│   ├── feature_extraction.py
│   ├── model_cnn.py
│   ├── model_knn.py
│   └── train.py
│
├── results/
│   ├── cnn_confusion_matrix.png
│   ├── knn_confusion_matrix.png
│   └── metrics.json
│
└── README.md
```

##  Dataset: GTZAN
This project uses the **GTZAN Music Genre Dataset**, one of the most widely used datasets in MIR research.

- **1000 audio files**  
- **10 genres**  
- **30 seconds per track**  
- `.wav` format
- https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

##  Feature Extraction (MFCC)
MFCC extraction steps include:

- Pre-emphasis  
- Frame segmentation  
- Windowing  
- FFT  
- Mel filterbank  
- Log compression  
- DCT  
- **13 MFCC coefficients per frame**

##  Models Used

###  1. Convolutional Neural Network (CNN)
Architecture:
- Conv2D(32) → MaxPool → Dropout  
- Conv2D(64) → MaxPool → Dropout  
- Conv2D(128) → MaxPool → Dropout  
- Dense(128) → Dropout  
- Softmax output (10 genres)

###  2. K-Nearest Neighbors (KNN)
- k = 5  
- Normalized features  
- GridSearchCV for tuning  
- Euclidean & Manhattan distances  

##  Results

### **CNN**
- Train Accuracy: **98.43%**  
- Val Accuracy: ~70%  

### **KNN**
- Train Accuracy: **81%**  
- Test Accuracy: **73%**  
- Best K (Hyperparameter) : 5  

## How to Run

```
git clone https://github.com/VISHAL-0805/Music_genre_classification.git
cd Music_genre_classification
pip install -r requirements.txt
jupyter notebook notebooks/research_paper_updated.ipynb
```

##  Open in Google Colab
https://colab.research.google.com/drive/1aYPfoWSbfSDYmNC4UqlwJPJyosD8ck3S?usp=sharing

##  Citation
```
Vishal Singh, Pushker Jain, Ayan Sar, Tanupriya Chowdhury, Ketan Kotecha.
"Differentiation of Music Genre from an Audio File Using Neural Networks."
2025.
```

## 📘 License
MIT License
