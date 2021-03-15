# MRP_MoCo

## Residual learning for motion correction in 3D multiparametric MRI

    Overview
    System Requirements and installation
    Reconstruction pipeline and motion simulation
    
### Overview

This repository contains a tensorflow implementation of a 3D patch-based multiscale CNN to resolve artifacts in 3D multiparametric MRI due to continuous motion that 
are not captured by prior navigator-based correction (Kurzawski et al, MRM, 2020).
The CNN takes the T1, T2 and proton density maps as obtained after navigator-based realignment and learns the residual deviation from the high-quality, 
motion-free reference.  

### System Requirements and installation

The code was developed with Python3.5, cuda 10.1 and tensorflow-gpu==1.12.0. All other dependencies are listed in the requirements.txt file. 

### Reconstruction pipeline and motion simulation

The CNN-based motion correction was demonstrated for the 3D Quantitative Transient-state Imaging (QTI) technique (Gómez et al, SciRep, 2020) and builds on the 3D QTI
reconstruction pipeline. 

The recon_q reconstruction pipeline with integrated navigator-based correction and motion simulation (for CNN training) are available on reasonable request. 
