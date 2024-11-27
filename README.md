# Implementation and Analysis of a Simple Deep Learning Technique for Single Image De-raining

# Problem Statement
Rain streaks significantly degrade the visual quality of images and negatively impact computer vision tasks such as object detection, security analysis, and autonomous driving. These applications rely on high-quality visuals for accurate decision-making and performance. For instance:

- In autonomous driving, rain can obscure road signs, vehicles, and pedestrians, posing safety risks.
- In security systems, rain-streaked images can hinder the identification of critical details needed for monitoring.

Addressing this problem is challenging due to the varying size, shape, and intensity of rain streaks, which require fine-grained image processing techniques. Existing methods are often computationally expensive and rely heavily on synthetic datasets, limiting their applicability to real-world rainy conditions. Additionally, most approaches focus on either real or synthetic datasets, which reduces their generalizability across diverse scenarios.

# Proposed Solution
To tackle these challenges, we developed a Convolutional Neural Network (CNN) model based on the U-Net architecture, enhanced with a self-supervised training method. Key aspects of our approach include:

- U-Net Backbone:
  U-Net is chosen for its encoder-decoder structure, which effectively captures both high- and low-level features.

- Self-Supervised Learning (SSL):
  Pretext-based SSL tasks allow the model to learn general feature representations, reducing computational load during the de-raining phase.

- Real Rain Dataset Training:
  Our model is trained on real rain datasets (RainDS and GTrain), making it better suited for real-world applications.

    
# Performance Metrics: 
To evaluate our model, we used the following metrics:

PSNR (Peak Signal-to-Noise Ratio)
SSIM (Structural Similarity Index Measure)

# Key Results
Final performance metrics for our model are as follows:

Training Results:

Mean Squared Error (MSE): 0.00688
PSNR: 27.61749
SSIM: 0.8374
Loss: 0.00684

Validation Results:

MSE: 0.05347
PSNR: 18.73917
SSIM: 0.6072
Loss: 0.07059

We achieved these results using the RainDS and GTrain datasets. Initially, we explored using a dashcam dataset, but the rain streaks were unclear, creating biases in the training data. Therefore, we limited our training to the RainDS and GTrain datasets, which provided better consistency and clarity.


# Model Block Diagram:

Our model uses a U-Net architecture with a pre-trained encoder, enhanced through a self-supervised learning (SSL) rotational classification task. The encoder extracts general features, while the decoder reconstructs rain-free images using upsampling and skip connections to retain spatial information. This workflow balances efficiency and performance for rain streak removal.

![Group 10](https://github.com/user-attachments/assets/595274c0-af7c-41cf-8bcc-6aa02b7bcfa3)


# Datasets used:
- RaindDS- https://drive.google.com/file/d/12yN6avKi4Tkrnqa3sMUmyyf4FET9npOT/view
- GTRain- https://drive.google.com/drive/folders/1NSRl954QPcGIgoyJa_VjQwh_gEaHWPb8

# Self-Supervised Learning (SSL) - Reference materials:
- [Self-Supervised Representation Learning by Rotation Feature Decoupling](https://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.pdf)
- [Review — RotNet: Unsupervised Representation Learning by Predicting Image Rotations](https://sh-tsang.medium.com/review-rotnet-unsupervised-representation-learning-by-predicting-image-rotations-60f4e4f3cf67)

# Pytorch - reference:
- [A PyTorch Tools, best practices & Styleguide](https://github.com/IgorSusmelj/pytorch-styleguide)
