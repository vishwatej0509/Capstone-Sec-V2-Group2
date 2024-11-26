import streamlit as st
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import base64

# Set page configuration as the very first Streamlit command
st.set_page_config(layout="wide", page_title="Image Deraining")

# Function to set the local background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .main {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background image
set_background("splash.jpg")  # Replace with your local image file

# Model and transformations
stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*stats, inplace=True),
    transforms.Resize((224, 224))
])

# UNet Model Definition
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(3, 64)
        self.contracting_12 = nn.MaxPool2d(2, 2)
        self.contracting_21 = self.conv_block(64, 128)
        self.contracting_22 = nn.MaxPool2d(2, 2)
        self.contracting_31 = self.conv_block(128, 256)
        self.contracting_32 = nn.MaxPool2d(2, 2)
        self.contracting_41 = self.conv_block(256, 512)
        self.contracting_42 = nn.MaxPool2d(2, 2)
        self.middle = self.conv_block(512, 1024)
        self.expansive_11 = nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1)
        self.expansive_12 = self.conv_block(1024, 512)
        self.expansive_21 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.expansive_22 = self.conv_block(512, 256)
        self.expansive_31 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.expansive_32 = self.conv_block(256, 128)
        self.expansive_41 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.expansive_42 = self.conv_block(128, 64)
        self.output = nn.Conv2d(64, num_classes, 3, 1, 1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, X):
        c11 = self.contracting_11(X)
        c12 = self.contracting_12(c11)
        c21 = self.contracting_21(c12)
        c22 = self.contracting_22(c21)
        c31 = self.contracting_31(c22)
        c32 = self.contracting_32(c31)
        c41 = self.contracting_41(c32)
        c42 = self.contracting_42(c41)
        m = self.middle(c42)
        e11 = self.expansive_11(m)
        e12 = self.expansive_12(torch.cat((e11, c41), 1))
        e21 = self.expansive_21(e12)
        e22 = self.expansive_22(torch.cat((e21, c31), 1))
        e31 = self.expansive_31(e22)
        e32 = self.expansive_32(torch.cat((e31, c21), 1))
        e41 = self.expansive_41(e32)
        e42 = self.expansive_42(torch.cat((e41, c11), 1))
        return self.output(e42)

# Load model
model = UNet(num_classes=3)
model_weights = torch.load("best_model.pt", map_location=torch.device('cpu'))
model.load_state_dict(model_weights, strict=False)

def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def resize_to_match(image, target_height, target_width):
    """
    Resize an image to match the target height and width.
    """
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def generate_and_display_images(uploaded_file):
    if uploaded_file is not None:
        # Load input image
        img = Image.open(uploaded_file).convert("RGB")
        img.save("input_image.jpg")

        # Keep original input image for display
        org_img = np.array(img)

        # Transform input image for model processing
        src_img = image_transforms(org_img).unsqueeze(0)

        # Generate output image
        gen_img = model(src_img)
        gen_img = denormalize(gen_img, *stats).squeeze().permute(1, 2, 0).detach().numpy()
        gen_img = np.clip((gen_img - gen_img.min()) / (gen_img.max() - gen_img.min()), 0, 1)

        # Resize the output image to match the input image dimensions
        gen_img_resized = resize_to_match(
            (gen_img * 255).astype(np.uint8),  # Scale gen_img to 0-255 for display
            target_height=org_img.shape[0],
            target_width=org_img.shape[1]
        )

        # Display input and output images side by side
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.image(org_img, caption="Input Image", use_column_width=True)
        with col2:
            st.image(gen_img_resized, caption="Derained Image", use_column_width=True)

if __name__ == "__main__":
    st.title("🌧️ Image Deraining Application")
    st.markdown("#### Upload a rainy image and make it clear! 🎨")
    uploaded_file = st.file_uploader("Upload your file here...", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        if st.button("Generate Derained Image"):
            generate_and_display_images(uploaded_file)




