AIEchoDx
=====================================

### News
6 May 2019:
* Upload AIEchoDx_demo and AIEchoDx_demo_notebook

### Introduction
AIEchoDx, which stands for “Artificial Intelligence Echocardiogram Diagnosis Network”, is a two-stage neural network to diagnose patients with atrial septal defect (ASD), dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), prior myocardial infarction (prior MI) from normal controls. It reads 45 frames of echocardiographic videos to make the diagnosis.

### Citations
Paper is under review

### Table of Contents
* [1. Installation and Requirements](#1-installation-and-requirements)

### 1. Installation and Requirements
```
python version = 3.6.5
tensorflow version = 1.10.0
keras version = 2.2.2
```

1.1 Clone the repository
```
git clone https://github.com/superresolution/AIEchoDx.git
```
1.2 setup a new conda enviroment
```
conda create --name aiechodx --file requirement
```
