AIEchoDx
=====================================

### News
6 May 2019:
* Upload AIEchoDx_demo and AIEchoDx_demo_notebook

### Introduction
AIEchoDx, which stands for “Artificial Intelligence Echocardiogram Diagnosis Network”, is a two-stage neural network to diagnose patients with atrial septal defect (ASD), dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), prior myocardial infarction (prior MI), and normal controls. It reads 45 frames of echocardiographic videos to make the diagnosis.

### Citations
Paper is under review

### Table of Contents
* [1. Installation and Requirements](#1-installation-and-requirements)
  * [1.1. Clone the repository](#11-clone-the-repository)
  * [1.2. Setup a new conda enviroment](#12-setup-a-new-conda-enviroment)
  * [1.3. Required Data Pre-Processing](#13-required-data-pre-processing)
* [2. Training AIEchoDx network](#2-training-aiechodx-network)
  * [2.1. Train the Google's Inception V3 network](#21-train-the-Google's-inception-v3-network)
  * [2.2. Transfer a 45 frame-echo videos into a 45×2048 matrix](#22-transfer-a-45-frame-echo-videos-into-a-45×2048-matrix)
  * [2.3. Required Data Pre-Processing](#23-required-data-pre-processing)
* [3. Testing](#3-testing)
* [4. Compare with human physicians](#4-compare-with-human-physicians)
* [5. CAM analysis](#5-cam-analysis)
* [6. DCM patients clustering](#6-dcm-patients-clustering)


### 1. Installation and Requirements
System Requirements: Windows (>= 7), Mac OS X (>= 10.8) or Linux
All other software dependencies are listed below:
```
python version = 3.6.5
tensorflow version = 1.10.0
keras version = 2.2.2
cv2 = 3.1.0
PIL = 5.2.0
pydicom = 1.2.2
seaborn = 0.9.0
```

#### 1.1 Clone the repository
Download software from github and cd to 
```
git clone https://github.com/superresolution/AIEchoDx.git
cd ~/AIEchoDx
```
#### 1.2 Setup a new conda enviroment
Make and activate a new Python 3.6 environment
##### Installation with conda code
```
conda env create -f requirement.yml
```
##### Installation with .sh script
```
./setup.sh
```
After installation, activate aiechodx enviroment:
```
conda activate aiechodx
```
This enviroment could be deactivate by:
```
conda deactivate
```
#### 1.3 Required Data Pre-Processing
The echocardiographic images need pre-preocessing before training in our AIEchoDx network. Here we provided seveal useful python functions to read, write and optimize echocardiographic images in `util.py`.
In detals, the python functions `load_dcm_video` and `load_avi_video` are used to read and transfer `.dcm` and `.avi` format videos into numpy matrix. The python function `load_dcm_information` is used to read patients informations. The function `limited_equalize` is used to optimize the contracts of medical images. The functions `remove_info` and `remove_info2` are used to remove the information of patients. 
Othervise, echocardiographic images need to be cut and save into smaller `.png` format before training.

### 2. Training AIEchoDx network
AIEchoDx is a two-stage network. We train the network separatedly. But in the future, we can combine the two stage networks and train them togather.
#### 2.1 Train the Google's Inception V3 network

#### 2.2 Transfer a 45 frame-echo videos into a 45×2048 matrix

#### 2.3 Train the diagnostic network

### 3. Testing

### 4. Compare with human physicians

### 5. CAM analysis

### 6. DCM patients clustering
