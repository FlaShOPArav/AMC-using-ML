# Automatic Modulation Classification using CNN

---

##  Project Overview

Automatic Modulation Classification (AMC) is a critical function in modern wireless communication systems, enabling intelligent receivers to identify the modulation scheme of a received signal without prior knowledge of the transmitter.  

This project implements a **Convolutional Neural Network (CNN)**–based approach for AMC using **raw in-phase and quadrature (IQ) samples**. The proposed model learns modulation-specific features directly from time-domain signals, eliminating the need for handcrafted feature extraction techniques.

---

##  Objectives

- To design and implement a CNN-based model for automatic modulation classification  
- To train the model using real-world radio signals from the RadioML dataset  
- To analyze training performance using epoch-wise loss reduction  
- To demonstrate the effectiveness of deep learning in communication signal processing  

---

## Dataset

- **Dataset:** RadioML 2016.10a  
- **Data Type:** Raw IQ samples  
- **Sample Shape:** (2 × 128)  
- **Modulation Schemes:** BPSK, QPSK, 8PSK, QAM16, QAM64, AM, FM, etc.  
- **SNR Range:** −20 dB to +18 dB  

*Due to large file size, the dataset is not included in this repository and must be downloaded separately.*

---

## Methodology

1. **Data Preprocessing**
   - Loading raw IQ samples from the RadioML dataset
   - Reshaping signals to match CNN input format

2. **CNN Architecture**
   - One-dimensional convolutional layers for feature extraction
   - ReLU activation functions
   - Max pooling for dimensionality reduction
   - Fully connected layers for multi-class classification

3. **Training Strategy**
   - Loss Function: Cross-Entropy Loss
   - Optimizer: Adam
   - Batch-wise training over multiple epochs
   - Monitoring epoch-wise loss values

4. **Evaluation**
   - Analysis of training convergence using loss reduction
   - Verification of learning behaviour across epochs

---


##  Technologies Used

- **Programming Language:** Python 3.10  
- **Deep Learning Framework:** PyTorch  
- **Libraries:** NumPy, Torch  
- **Development Tools:** VS Code, macOS  

---

##  How to Run the Project

1. Create and activate virtual environment:
```bash
python3 -m venv amc_env
source amc_env/bin/activate
```

##Install dependencies:
pip install -r requirements.txt

##Run the training script:
python -m src.train.train

## High CPU utilization during training is expected

## Results
The CNN successfully trains on raw IQ samples
Epoch-wise loss decreases, indicating effective learning
The model demonstrates feasibility of deep learning–based AMC

## Applications
Cognitive Radio Systems
Software Defined Radio (SDR)
Spectrum Monitoring
Intelligent Wireless Receivers
Defence and Secure Communication

##Author
-Arav Upadhyay
