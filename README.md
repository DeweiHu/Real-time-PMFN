# Real-time-PMFN
real-time OCT denoising implemented on SECTR imaging system

![pipeline](../master/Real-time-PMFN/imgs/PMFN.png)

## Main
The main function applys the whole PMFN processing pipeline

(1) Motion correction

(2) Create pseudo-modality [network mimic of self-fusion]

(3) Get edge map from pseudo-modality by sobel kernel

(4) Denoising network

## src
The source functions are

(1) MotionCorrection

(2) util

## Model

(1) The network architechture

(2) trained model
