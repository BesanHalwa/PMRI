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

