# Lower Limb Calf Muscle Segmentation from Diffusion-Weighted Magnetic Resonance Images Using Deep Learning

Peripheral artery disease (PAD) affects blood flow
to the limbs, and diffusion-weighted magnetic resonance imaging
(DW-MRI) can quantify microvascular perfusion and diffusion
in calf muscles, aiding diagnosis. However, manual segmentation
is time consuming and subjective. We propose a conditional
generative adversarial network (cGAN) with an enhanced U-Net
architecture for automated segmentation of calf muscles from
DW-MRI. Our method leverages data augmentation to address
small dataset sizes, splitting images into left and right halves and
applying flipping and progressive rotation. Evaluated on datasets
of healthy and PAD patients, our approach achieves average Dice
Similarity Coefficient (Dice) scores of 54.86% to 79.85% across
muscle groups, significantly outperforming baseline models
(original U-Net architecture with cGAN and no data
augmentation). This work demonstrates the potential of deep
learning for automating segmentation in PAD diagnosis, offering
a scalable solution for clinical applications.

![Diffusion-weighted MR image (left), manually delineated ROIs (centre), manually delineated ROIs overlaid on the DWI (right) of one subject](./images/des.png)

*Figure: Diffusion-weighted MR image (left), manually delineated ROIs (centre), manually delineated ROIs overlaid on the DWI (right) of one subject*

The enhanced U-Net adds forward blocks with skip connections after each upsampling and downsampling layer, preserving spatial information by not downsampling below 8x8 (unlike the original U-Net’s 1x1). This reduces parameters (159.38 MB vs. 207.62 MB) while improving generalization. Forward block is an additional convolutional block structurally identical to the preceding convolution layer. 
### Table below shows a comparison of the original U-Net and the proposed Enhanced U-Net.

| Original U-Net        |                       |           | Enhanced U-Net        |                       |           |
|-----------------------|-----------------------|-----------|-----------------------|-----------------------|-----------|
| Layer                 | Output Shape          | Params    | Layer                 | Output Shape          | Params    |
| Input Layer           | (None, 256, 256, 3)   | 0         | Input Layer           | (None, 256, 256, 1)   | 0         |
| Sequential_2          | (None, 128, 128, 64)  | 3,072     | Sequential_2          | (None, 128, 128, 64)  | 1,280     |
| Sequential_3          | (None, 64, 64, 128)   | 131,584   | Sequential_3          | (None, 128, 128, 64)  | 65,792    |
| Sequential_4          | (None, 32, 32, 256)   | 525,312   | Sequential_4          | (None, 64, 64, 128)   | 131,584   |
| Sequential_5          | (None, 16, 16, 512)   | 2,099,200 | Sequential_5          | (None, 64, 64, 128)   | 262,656   |
| Sequential_6          | (None, 8, 8, 512)     | 4,196,352 | Sequential_6          | (None, 32, 32, 256)   | 525,312   |
| Sequential_7          | (None, 4, 4, 512)     | 4,196,352 | Sequential_7          | (None, 32, 32, 256)   | 1,049,600 |
| Sequential_8          | (None, 2, 2, 512)     | 4,196,352 | Sequential_8          | (None, 16, 16, 512)   | 2,099,200 |
| Sequential_9          | (None, 1, 1, 512)     | 4,196,352 | Sequential_9          | (None, 16, 16, 512)   | 4,196,352 |
| Sequential_10         | (None, 2, 2, 512)     | 4,196,352 | Sequential_19         | (None, 8, 8, 1024)    | 8,392,704 |
| Concatenate           | (None, 2, 2, 1024)    | 0         | Sequential_11         | (None, 16, 16, 512)   | 8,390,656 |
| Sequential_11         | (None, 4, 4, 512)     | 8,390,656 | Concatenate           | (None, 16, 16, 1024)  | 0         |
| Concatenate_1         | (None, 4, 4, 1024)    | 0         | Sequential_12         | (None, 16, 16, 512)   | 8,390,656 |
| Sequential_12         | (None, 8, 8, 512)     | 8,390,656 | Concatenate_1         | (None, 16, 16, 1024)  | 0         |
| Concatenate_2         | (None, 8, 8, 1024)    | 0         | Sequential_13         | (None, 32, 32, 256)   | 4,195,328 |
| Sequential_13         | (None, 16, 16, 512)   | 8,390,656 | Concatenate_2         | (None, 32, 32, 512)   | 0         |
| Concatenate_3         | (None, 16, 16, 1024)  | 0         | Sequential_14         | (None, 32, 32, 256)   | 2,098,176 |
| Sequential_14         | (None, 32, 32, 256)   | 4,195,328 | Concatenate_3         | (None, 32, 32, 512)   | 0         |
| Concatenate_4         | (None, 32, 32, 512)   | 0         | Sequential_15         | (None, 64, 64, 128)   | 1,049,088 |
| Sequential_15         | (None, 64, 64, 128)   | 1,049,088 | Concatenate_4         | (None, 64, 64, 256)   | 0         |
| Concatenate_5         | (None, 64, 64, 256)   | 0         | Sequential_16         | (None, 64, 64, 128)   | 524,800   |
| Sequential_16         | (None, 128, 128, 64)  | 262,400   | Concatenate_5         | (None, 64, 64, 256)   | 0         |
| Concatenate_6         | (None, 128, 128, 128) | 0         | Sequential_17         | (None, 128, 128, 64)  | 262,400   |
| Conv2DTranspose_8     | (None, 256, 256, 3)   | 6,147     | Concatenate_6         | (None, 128, 128, 128) | 0         |
|                       |                       |           | Sequential_18         | (None, 128, 128, 64)  | 131,328   |
|                       |                       |           | Concatenate_7         | (None, 128, 128, 128) | 0         |
|                       |                       |           | Conv2DTranspose_8     | (None, 256, 256, 6)   | 12,294    |
| Total params:         | 54,425,859            | 207.62 MB | Total params:         | 41,779,206            | 159.38 MB |
| Trainable params:     | 54,414,979            | 207.58 MB | Trainable params:     | 41,769,478            | 159.34 MB |
| Non-trainable params: | 10,880                | 42.50 KB  | Non-trainable params: | 9,728                 | 38.00 KB  |                                                                   


### Training configuration of various models
| MODELS                 | Discriminator  | Generator      | Training Data    | Data Augmentation | Fine tuning |
|------------------------|----------------|----------------|------------------|-------------------|-------------|
| SegModel_01 (Baseline) | Yes            | Original U-Net | Healthy Subjects | No                | No          |
| SegModel_02 (Baseline) | Yes            | Original U-Net | All Subjects     | No                | No          |
| SegModel_03            | Yes            | Enhanced U-Net | Healthy Subjects | No                | No          |
| SegModel_04            | Yes            | Enhanced U-Net | All Subjects     | No                | No          |
| SegModel_05            | Yes            | Enhanced U-Net | Healthy Subjects | Yes               | No          |
| SegModel_06            | No             | Enhanced U-Net | Healthy Subjects | Yes               | Yes         |
| SegModel_07            | No             | Enhanced U-Net | All Subjects     | Yes               | Yes         |


### Average Dice score for each muscle group by various segmentation models

| MODELS                       | TA left | PER left | DP left | SOL GL Left | GM left | TA right | PER right | DP right | SOL GL right | GM right |
|------------------------------|---------|----------|---------|-------------|---------|----------|-----------|----------|--------------|----------|
| SegModel_01 (baseline model) | 29.32%  | 4.09%    | 1.01%   | 4.26%       | 2.65%   | 29.23%   | 4.22%     | 1.61%    | 5.11%        | 1.88%    |
| SegModel_02 (baseline model) | 2.27%   | 1.21%    | 2.86%   | 0.00%       | 0.00%   | 10.88%   | 2.07%     | 1.74%    | 0.00%        | 0.00%    |
| SegModel_03                  | 1.98%   | 0.14%    | 0.21%   | 1.24%       | 0.19%   | 0.89%    | 0.08%     | 1.95%    | 1.95%        | 0.22%    |
| SegModel_04                  | 9.03%   | 2.40%    | 4.20%   | 0.00%       | 7.21%   | 7.68%    | 3.21%     | 3.83%    | 0.16%        | 7.65%    |
| SegModel_05                  | 62.42%  | 34.81%   | 36.12%  | 65.93%      | 47.47%  | 67.86%   | 44.98%    | 42.99%   | 70.95%       | 48.42%   |
| SegModel_06                  | 38.54%  | 27.48%   | 51.71%  | 67.37%      | 45.84%  | 69.57%   | 56.52%    | 51.65%   | 71.61%       | 50.18%   |
| SegModel_07                  | 66.89%  | 54.86%   | 66.24%  | 77.82%      | 57.95%  | 79.85%   | 74.27%    | 62.83%   | 78.04%       | 60.54%   |

![Visual comparison of the manually drawn region of interest and auto segmentation (predicted region of interest) using model SegModel_07](./images/results.png)

*Figure: Visual comparison of the manually drawn region of interest and auto segmentation (predicted region of interest) using model SegModel_07*

### Model Weighths (Google Drive)
https://drive.google.com/drive/folders/14lrvQqbxoKXOvp9Io2NRLoSMofUqsBBj?usp=sharing 

### Some additional results (generated ROIs)
https://drive.google.com/drive/folders/1Jd3pl1nu_JNFjll9j3ZyP9SHozF3qy73?usp=sharing 

### Some additional notebooks
https://drive.google.com/drive/folders/1hR-QajvbmsXuIOGX4qkG6hW6z94UDexx?usp=sharing 

### Data
The data used for this project is proprietary and not publicly available.


