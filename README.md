# Facial Recognition System with Anti-spoofing Measures and Liveness Detection Models.
Bachelor's degree graduation project that aims to find a better solution to prevent replay attacks on face recognition system
## Team members
1. Ahmed Mohamed AbdelRashied ahmedm.abdelrashied@gmail.com
2. Ahmed Rushdi Mohammed 
3. Aoss Majed Sultan Zaid 
4. Eslam Nasser Abdelqader 
5. Mohammed Walid Moahmmed 
6. Moahmmed ElSayed AbdelHamid

## Datasets
1. CASIA-MFSD
   is a confidential dataset containing videos for the team members combined with videos collected from YouTube by us to meet our case 
2. Our Dataset
   is a dataset for face anti-spoofing. , and 12 videos for each subject under different resolutions and light conditions. Three various spoof attacks are designed: replay, warp print and cut print attacks.
* Experimentation
* System Overview
## Experimentation
![Expermimentation](https://github.com/Ahmed-Rushdi/liveness_system/tree/master/Trainig_preprocessing_misc)
 with the different liveness detection models in [1], [2], [3] and Volumetric CNN.

## System Overview

>Models Folder: https://drive.google.com/drive/folders/1DMfr07Z017H3F0hwy_vZpZo-QGQwd7is?usp=sharing

We combine prompt-based authentication with liveness detection in a ![client](https://github.com/Ahmed-Rushdi/liveness_system/tree/master/Client)-![server](https://github.com/Ahmed-Rushdi/liveness_system/tree/master/server) environment.

Prompts are {'left', 'right', 'smile'}

![plot](https://github.com/Ahmed-Rushdi/liveness_system/blob/e907cec8c84fcb1c7b7efebaff9476913b110372/image.png)

## Refrences 
[1] Y. Moon and I. Ryoo and S. Kim "Face Antispoofing Method Using Color Texture Segmentation on FPGA,"Hindawi Security and Communication Networks, Vol, 2021, Article ID 9939232, 11 pages,2021.

[2] Rehman, Y., Po, L. and Liu, M., 2020. SLNet: Stereo face liveness detection via dynamic disparity-maps and convolutional neural network. Expert Systems with Applications, 142, p.113002.

[3] Lakshminarayana, N.N., Narayan, N., Napp, N., Setlur, S. and Govindaraju, V., 2017, February. A discriminative spatio-temporal mapping of face for liveness detection. In 2017 IEEE International Conference on Identity, Security and Behavior Analysis (ISBA) (pp. 1-7). IEEE.
