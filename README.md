# Real-time-PMFN
real-time OCT denoising implemented on SECTR imaging system

<p align="center">
  <img src="/imgs/PMFN.png" width="850" title="PMFN pipeline">
</p>

## Training of Nsf
The model Nsf is used to map the high noise input <img src="https://render.githubusercontent.com/render/math?math=X_{i}"> to the self-fusion of the corresponding low noise image <img src="https://render.githubusercontent.com/render/math?math=S_{i}">. In /src/Nsf_train/, all the pre-processing and model training code is included. 

(1) train_data_prep.py

This file is used to get n-frame average <img src="https://render.githubusercontent.com/render/math?math=Y_{i}"> by (a) shown in the pipleline. Other than this, it register r-neighboring bscans of <img src="https://render.githubusercontent.com/render/math?math=X_{i}"> and save it as a stack of (2r+1) channels for later self-fusion. 

## Model

(1) The network architechture

(2) trained model
