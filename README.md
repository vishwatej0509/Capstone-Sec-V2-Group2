# Implementation and Analysis of a Simple Deep Learning Technique for Single Image De-raining

The problem we are addressing in this work is the removal of rain streaks from images,
whether these streaks are caused by real or synthetic rain. Rain streaks significantly degrade
the visual quality of images and can negatively affect critical computer vision tasks. These
include tasks such as object detection, security system analysis, and autonomous driving,
where clear image quality is crucial for decision-making and performance.

This problem is especially important because rain streaks blur or obscure key visual information that is needed to make accurate decisions in these applications. For example, in
autonomous driving, rain can hinder the detection of road signs, vehicles, and pedestrians,
posing safety risks. Similarly, rain-streaked images in surveillance systems may obscure
critical details necessary for security monitoring. Therefore, improving the clarity of such
images has a direct impact on safety, security, and performance in real-world scenarios.
Single Image de-raining is a challenging problem because rain streaks vary widely in size,
shape, and intensity. Additionally, detecting rain streaks in a single image requires finegrained image processing techniques that can distinguish rain from other image features.
Most existing solutions are computationally intensive and often rely on synthetic datasets,
which may not generalize well to real-world rainy conditions. Moreover, previous methods typically either focus on real or synthetic rain datasets separately, which limits their
applicability across various scenarios.


To address this issue, we will be developing a Convolutional Neural Network (CNN) model
based on the U-Net architecture. Our approach incorporates a self-supervised training
method that allows the model to learn general features during a pretext-based task, thereby
reducing the computational load on U-Net during the rain streak removal phase. By using real rain datasets for training, our model aims to improve performance on real-world
applications by generating clearer, de-rained images.
Our model will be evaluated using performance metrics like PSNR, SSIM, BRISQUE, and
NIQE to ensure it generates sharp and quality images, and we will also compare it with
state-of-the-art models.

Rain streak removal will be greatly helpful for areas like autonomous driving, security
surveillance, and object detection systems, where high-quality visuals are essential for accurate detection and decision-making. By improving the clarity of images impacted by rain,
our model will enhance the performance of these systems, contributing to safer and more
reliable operations in challenging weather conditions.

# Model Block Diagram(Midterm):
UNet with Encoder followed by classfication. Decoder part is masked as it will be implemented after midterm.

![Model_Block diagram](https://github.com/user-attachments/assets/b0a9ac3b-7780-4368-87d1-0a72a14e51e3)


# Prospective datasets:
- Stereo Image Dataset- https://deepblue.lib.umich.edu/data/concern/data_sets/cc08hg37c?locale=en
- Deraindrop Dataset- https://drive.google.com/open?id=1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K

# Self-Supervised Learning (SSL) - Reference materials:
- [Self-Supervised Representation Learning by Rotation Feature Decoupling](https://openaccess.thecvf.com/content_CVPR_2019/papers/Feng_Self-Supervised_Representation_Learning_by_Rotation_Feature_Decoupling_CVPR_2019_paper.pdf)
- [Review — RotNet: Unsupervised Representation Learning by Predicting Image Rotations](https://sh-tsang.medium.com/review-rotnet-unsupervised-representation-learning-by-predicting-image-rotations-60f4e4f3cf67)

# Pytorch - reference:
- [A PyTorch Tools, best practices & Styleguide](https://github.com/IgorSusmelj/pytorch-styleguide)
