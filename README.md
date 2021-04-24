# Real-time-PMFN
real-time OCT denoising implemented on SECTR imaging system

<p align="center">
  <img src="/imgs/PMFN.png" width="850" title="PMFN pipeline">
</p>

## Training of Nsf
The model Nsf is used to map the high noise input <img src="https://render.githubusercontent.com/render/math?math=X_{i}"> to the self-fusion of the corresponding low noise image <img src="https://render.githubusercontent.com/render/math?math=S_{i}">. In /src/Nsf_train/, all the pre-processing and model training code is included. 

(1) train_data_prep.py

This file is used to get n-frame average <img src="https://render.githubusercontent.com/render/math?math=Y_{i}"> by (a) shown in the pipleline. Also, the 1st frame <img src="https://render.githubusercontent.com/render/math?math=X^{1}"> of the n repeats is extracted. Other than this, it register r-neighboring bscans of <img src="https://render.githubusercontent.com/render/math?math=Y_{i}"> and save it as a stack of (2r+1) channels for later self-fusion. In conclusion, there are 3 outputs for each raw volume, X,Y (Nifiti) and a pickle file that contains the low noise (Y) bscans stacks. 

(2) Get_SFy.py

Given the aligned r-neighborhood of <img src="https://render.githubusercontent.com/render/math?math=Y_{i}">, the self-fusion is applied to get <img src="https://render.githubusercontent.com/render/math?math=S_{i}">. 


## Model

(1) The network architechture

(2) trained model
