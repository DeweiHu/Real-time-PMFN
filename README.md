# Real-time-PMFN
real-time OCT denoising implemented on SECTR imaging system

<p align="center">
  <img src="/imgs/PMFN.png" width="850" title="PMFN pipeline">
</p>

## Utilities
Under /utils/ there are some functions used. **util.py** includes the Nifiti file reader/writer, intensity normalizer etc. **MotionCorrection.py** is the function used for rigid registration. **label_fusion** is joint label fusion sofware. **self_fusion.sh** is a shell script that inplement the label fusion software. There are 3 directories to be specified, dir of label_fusion, dir of c3d, dir of atlases (bscans in our case).

## Training of Nsf
The model Nsf is used to map the high noise input <img src="https://render.githubusercontent.com/render/math?math=X_{i}"> to the self-fusion of the corresponding low noise image <img src="https://render.githubusercontent.com/render/math?math=S_{i}">. In /src/Nsf_train/, all the pre-processing and model training code is included. 

**(1) train_data_prep.py**

This file is used to get n-frame average <img src="https://render.githubusercontent.com/render/math?math=Y_{i}"> by (a) shown in the pipleline. Also, the 1st frame <img src="https://render.githubusercontent.com/render/math?math=X^{1}"> of the n repeats is extracted. Other than this, it register r-neighboring bscans of <img src="https://render.githubusercontent.com/render/math?math=Y_{i}"> and save it as a stack of (2r+1) channels for later self-fusion. In conclusion, there are 3 outputs for each raw volume, X,Y (Nifiti) and a pickle file that contains the low noise (Y) bscans stacks. 

**(2) Get_SFy.py**

Given the aligned r-neighborhood of <img src="https://render.githubusercontent.com/render/math?math=Y_{i}">, the self-fusion is applied to get <img src="https://render.githubusercontent.com/render/math?math=S_{i}">. 

**(3) PM_dataloader.py**

The PM (stands for pseudo-modality) dataloader is used to pair (<img src="https://render.githubusercontent.com/render/math?math=X_{i}">,<img src="https://render.githubusercontent.com/render/math?math=S_{i}">). In order to provide sufficient information for the model to mimic the self-fusion. The input has 3 channels including <img src="https://render.githubusercontent.com/render/math?math=X_{i-1}">,<img src="https://render.githubusercontent.com/render/math?math=X_{i}">,<img src="https://render.githubusercontent.com/render/math?math=X_{i+1}">. Adjacent bscans are also registered. The output will be a pickled list. Each element of the list will be a (X,S) pair.

**(4) PM_train.py**

The training of the model. The architecture of the model is the multi-scale U-Net which is available under /Models/NetworkArch.py

## Model

(1) The network architechture

(2) trained model
