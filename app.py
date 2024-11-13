import streamlit as st
import pickle
import streamlit as st
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from matplotlib import pyplot
import numpy as np
import cv2 
from PIL import Image
import PIL




stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats, inplace=True),
        transforms.Resize((224, 224), interpolation=PIL.Image.BILINEAR)
    ])

class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        return output_out

# model = torch.load("./image_deraining_v1.pt", map_location=torch.device('cpu'))
model = UNet(num_classes = 3)
model_weights = torch.load("image_deraining_v3.pt",  map_location=torch.device('cpu'))
model.load_state_dict(model_weights, strict = False)
# print(type(model))


def denormalize(images, means, stds):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means

def generate_and_display_images(uploaded_file):
    if uploaded_file is not None:
        # Saves
        img = Image.open(uploaded_file)
        img = img.save("img.jpg")

        org_img = cv2.imread("img.jpg")
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
        # st.image(uploaded_file)
        src_img = cv2.imread("img.jpg")
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_img = image_transforms(src_img)
        src_img = torch.unsqueeze(src_img, 0)
        
        print(f"Resized shape: {src_img.shape}")
        
        # generate image from source
        gen_img = model(src_img)
        # assert torch.equal(gen_img, src_img), "Both are same"

        gen_img = denormalize(gen_img, *stats)
        gen_img = torch.squeeze(gen_img, 0).permute(1,2,0).detach().numpy()
        gen_img = (gen_img - gen_img.min())/(gen_img.max() - gen_img.min())

        src_img = denormalize(src_img, *stats)
        src_img = torch.squeeze(src_img, 0).permute(1,2,0).detach().numpy()
        src_img = (src_img - src_img.min())/(src_img.max() - src_img.min())

        col1, col2 = st.columns(2, gap = "large")
        with col1:
            st.image(src_img)
            st.markdown("Input image")
        with col2:
            st.image(gen_img)
            st.markdown("Derained image")


if __name__ == "__main__":
    
    st.set_page_config(layout='centered',
                    page_title="Image deraining")
    
    st.header("Make a clear image")

    st.markdown("#")
    uploaded_file = st.file_uploader("Upload your file here...", type=['png', 'jpg', 'jpeg'])
    print(type(uploaded_file))
    # StreamLit application
    
    if st.button("Derain image..", use_container_width=True):
        generate_and_display_images(uploaded_file)


